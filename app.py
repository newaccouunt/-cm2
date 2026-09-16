import json
import os
from typing import Any

import duckdb
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
# 🔑 APNA HF TOKEN YAHAN DAALO
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# 📦 Dataset
HF_DATASET_REPO = os.environ.get(
    "ICMR_HF_REPO",
    "bronx-ultra/icrm-hitek-full-db-mixed-bucket"
)

# DuckDB hf:// paths
PHONE_FILES  = f"hf://datasets/{HF_DATASET_REPO}/*phone*.parquet"
AADHAR_FILES = f"hf://datasets/{HF_DATASET_REPO}/*aadhar*.parquet"
ALL_FILES    = f"hf://datasets/{HF_DATASET_REPO}/**/*.parquet"

DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]


# ── DuckDB Connection ───────────────────────────────────────────────────────
def _get_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar=false;")
    con.execute("SET http_keep_alive=false;")
    con.execute("SET http_timeout=25000;")
    if HF_TOKEN and HF_TOKEN.startswith("hf_"):
        con.execute(
            f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{HF_TOKEN}')"
        )
    return con


# ── Core Search ─────────────────────────────────────────────────────────────
def _duckdb_search(field: str, value: str, limit: int = 10) -> list[dict]:
    if field in ("phoneNumber", "otherNumber"):
        file_pattern, column = PHONE_FILES, field
    elif field == "aadharNumber":
        file_pattern, column = AADHAR_FILES, "aadharNumber"
    else:
        file_pattern, column = ALL_FILES, field

    try:
        con = _get_conn()
        query = f"""
            SELECT * FROM read_parquet('{file_pattern}', union_by_name=true)
            WHERE CAST({column} AS VARCHAR) = ?
            LIMIT ?
        """
        result = con.execute(query, [str(value).strip(), int(limit)]).fetchdf()
        con.close()

        if result.empty:
            return []
        return [
            {k: (None if str(v) == "nan" else v) for k, v in row.items()}
            for row in result.to_dict(orient="records")
        ]
    except Exception as e:
        print(f"⚠️ DuckDB error ({field}={value}): {e}")
        return []


# ── Connected Numbers ───────────────────────────────────────────────────────
def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value == "nan" or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected


def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen, out = {}, []
    for r in rows:
        ph = str(r.get("phoneNumber", "") or "").strip()
        ad = str(r.get("aadharNumber", "") or "").strip()
        key = (ph, ad) if (ph or ad) else (
            str(r.get("name", "")), str(r.get("fathersName", ""))
        )
        n = seen.get(key, 0)
        if n < DUPLICATE_CAP:
            seen[key] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out


# ── Unified Search ──────────────────────────────────────────────────────────
def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    if not q or not q.isdigit() or len(q) < 8:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    all_rows, searched = [], []

    if len(q) in (10, 11):
        rows = _duckdb_search("phoneNumber", q, limit)
        if rows:
            all_rows.extend(rows)
            searched.append("phoneNumber")

    if len(q) == 12:
        rows = _duckdb_search("aadharNumber", q, limit)
        if rows:
            all_rows.extend(rows)
            searched.append("aadharNumber")

    # Enrich — connected numbers se extra records
    if all_rows:
        connected_searches = set()
        for row in all_rows[:3]:
            for nf in NUMBER_FIELDS:
                nv = row.get(nf)
                if nv and str(nv) != "nan" and str(nv) not in connected_searches:
                    connected_searches.add(str(nv))
                    extra = _duckdb_search(nf, str(nv), limit=2)
                    if extra:
                        all_rows.extend(extra)
                        if nf not in searched:
                            searched.append(nf)

    all_rows = _cap_duplicates(all_rows)[:limit]

    return {
        "query": q,
        "searched_fields": searched,
        "count": len(all_rows),
        "results": all_rows,
    }


def _field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        return {
            "field": field, "value": value, "mode": mode,
            "count": 0, "results": [], "error": "Unknown field",
        }
    try:
        results = _cap_duplicates(_duckdb_search(field, value, limit))[:limit]
        return {
            "field": field, "value": value, "mode": mode,
            "count": len(results), "results": results,
        }
    except Exception as e:
        return {
            "field": field, "value": value, "mode": mode,
            "count": 0, "results": [], "error": str(e),
        }


# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(title="ICMR + HITEK Full Info API")


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


# ── UI ──────────────────────────────────────────────────────────────────────
UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICMR Full Info Search</title>
<style>
  *{box-sizing:border-box}
  body{font-family:-apple-system,system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px;background:#0a0a14;color:#e5e5ef}
  h1{font-size:1.5rem;margin-bottom:4px}
  .sub{color:#888;font-size:.85rem;margin-bottom:20px}
  .row{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
  input{flex:1;min-width:200px;padding:14px;font-size:16px;background:#15152a;color:#e5e5ef;border:1px solid #2a2a45;border-radius:10px;outline:none}
  input:focus{border-color:#6366f1}
  button{padding:14px 20px;font-size:16px;font-weight:600;background:#6366f1;color:#fff;border:none;border-radius:10px;cursor:pointer}
  button:hover{background:#4f46e5}
  button:disabled{background:#333}
  .result{background:#15152a;border:1px solid #2a2a45;border-radius:12px;padding:16px;margin-bottom:12px}
  .result h3{margin:0 0 10px;color:#a5b4fc;font-size:.95rem}
  .field{padding:5px 0;border-bottom:1px solid #1f1f35;font-size:.92rem}
  .field:last-child{border:none}
  .field b{color:#888;font-weight:500;min-width:120px;display:inline-block}
  .conn{margin-top:8px;padding:8px;background:#1e1b4b;border-radius:8px;font-size:.85rem}
  .conn b{color:#a5b4fc}
  .status{padding:10px 14px;border-radius:10px;margin-bottom:14px;background:#15152a;border:1px solid #2a2a45;font-size:.9rem}
  .err{background:#2a0f15;border-color:#7f1d1d;color:#fca5a5}
  .ok{background:#0f2a1a;border-color:#14532d;color:#86efac}
  .footer{text-align:center;color:#555;font-size:.8rem;margin-top:30px}
  .footer a{color:#6366f1;text-decoration:none}
</style>
</head>
<body>
  <h1>🔍 ICMR Full Info Search</h1>
  <div class="sub">Render.com pe hosted — phone ya aadhar daalo, pura info milega</div>
  <div class="row">
    <input id="q" placeholder="10-digit phone ya 12-digit aadhar..." autofocus>
    <button id="btn" onclick="go()">Search</button>
  </div>
  <div id="out"></div>
  <div class="footer">
    👨‍💻 <a href="https://t.me/api_wallah">@kzr0x</a> |
    📢 <a href="https://t.me/api_wallah">@api_wallah</a>
  </div>
<script>
const qEl=document.getElementById('q'),btnEl=document.getElementById('btn'),outEl=document.getElementById('out');
qEl.addEventListener('keypress',e=>{if(e.key==='Enter')go()});
async function go(){
  const q=qEl.value.trim(); if(!q)return;
  btnEl.disabled=true;
  outEl.innerHTML='<div class="status">⏳ Searching...</div>';
  try{
    const r=await fetch('/info/'+encodeURIComponent(q)+'?limit=10');
    const data=await r.json();
    render(data);
  }catch(e){
    outEl.innerHTML='<div class="status err">❌ '+e.message+'</div>';
  }finally{btnEl.disabled=false}
}
function render(data){
  if(!data.success||!data.results||!data.results.length){
    outEl.innerHTML='<div class="status err">❌ No data for <b>'+(data.number||'?')+'</b><br>Searched: '+((data.searched_fields||[]).join(', ')||'none')+'</div>';
    return;
  }
  let h='<div class="status ok">✅ Found <b>'+data.count+'</b> for <b>'+data.number+'</b> ('+data.number_type+')<br>Searched: '+(data.searched_fields||[]).join(', ')+'</div>';
  data.results.forEach((row,i)=>{
    h+='<div class="result"><h3>📄 Result #'+(i+1)+'</h3>';
    ['name','fathersName','phoneNumber','aadharNumber','otherNumber','address','district','pincode','state','town','source'].forEach(f=>{
      const v=row[f];
      if(v!=null&&String(v).trim()!==''&&String(v)!=='nan'){
        h+='<div class="field"><b>'+f+'</b>'+v+'</div>';
      }
    });
    if(row.connected_numbers&&row.connected_numbers.length){
      h+='<div class="conn"><b>🔗 Connected:</b> '+row.connected_numbers.map(c=>c.field+'=<b>'+c.value+'</b>').join(' | ')+'</div>';
    }
    h+='</div>';
  });
  outEl.innerHTML=h;
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": HF_DATASET_REPO,
        "runtime": "render",
        "token_loaded": bool(HF_TOKEN and HF_TOKEN.startswith("hf_")),
        "searchable_fields": ["phoneNumber", "aadharNumber"],
    }


@app.get("/debug")
def debug():
    try:
        con = _get_conn()
        result = con.execute(
            f"SELECT * FROM read_parquet('{PHONE_FILES}', union_by_name=true) LIMIT 1"
        ).fetchdf()
        con.close()
        return {
            "status": "connected",
            "dataset": HF_DATASET_REPO,
            "pattern": PHONE_FILES,
            "token_loaded": bool(HF_TOKEN and HF_TOKEN.startswith("hf_")),
            "columns": list(result.columns) if not result.empty else [],
        }
    except Exception as e:
        return {
            "status": "error",
            "dataset": HF_DATASET_REPO,
            "pattern": PHONE_FILES,
            "token_loaded": bool(HF_TOKEN and HF_TOKEN.startswith("hf_")),
            "error": str(e),
        }


@app.get("/search")
def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    aadhar: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=50),
    pretty: bool = Query(True),
):
    if aadhar:
        q_val, field = aadhar.strip(), "aadharNumber"
    elif mobile:
        q_val, field = mobile.strip(), "phoneNumber"
    elif q:
        q_val = q.strip()
    else:
        raise HTTPException(422, "Provide q, mobile, or aadhar")

    if not q_val:
        raise HTTPException(422, "Query cannot be empty")

    data = _field_search(field, q_val, mode, limit) if field else _unified_search(q_val, limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": q_val}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")


@app.get("/search/phone/{number}")
def search_phone(number: str, limit: int = Query(10, ge=1, le=50), pretty: bool = Query(True)):
    data = _field_search("phoneNumber", number, "exact", limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": number}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")


@app.get("/search/aadhar/{number}")
def search_aadhar(number: str, limit: int = Query(10, ge=1, le=50), pretty: bool = Query(True)):
    data = _field_search("aadharNumber", number, "exact", limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": number}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")


@app.get("/info/{number}")
def full_info(number: str, limit: int = Query(10, ge=1, le=50), pretty: bool = Query(True)):
    number = number.strip()
    data = _unified_search(number, limit)
    result = {
        "success": bool(data.get("count", 0) > 0),
        "number": number,
        "number_type": (
            "phone" if len(number) in (10, 11)
            else "aadhar" if len(number) == 12
            else "unknown"
        ),
        **data,
    }
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")


@app.post("/search/parallel")
def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 20:
        raise HTTPException(400, "max 20 queries")

    results = [
        _field_search(
            item.get("field", "phoneNumber"),
            item.get("value", ""),
            item.get("mode", "exact"),
            int(item.get("limit", req.limit)),
        )
        for item in req.queries
    ]
    return {"searches": len(req.queries), "results": results}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"\n🚀 ICMR Search API chal raha hai: http://0.0.0.0:{port}")
    print(f"📖 Docs: http://localhost:{port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
