import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import duckdb
import httpx
from huggingface_hub import HfFileSystem   # ✅ v1.6.0+ mein bucket support
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# 🔑 APNA HF TOKEN YAHAN DAALO (ya Render env var se set karo)
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# ✅ Bucket details
HF_BUCKET_ID = "bronx-ultra/icrm-hitek-full-db-mixed-bucket"
HF_BUCKET_DUCKDB = f"hf://buckets/{HF_BUCKET_ID}"

PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

IDX_PHONE = "idx_phone"
IDX_AADHAR = "idx_aadhar"

# ✅ SAARI 7 files use ho rahi hain — 100% index coverage
REMOTE_INDEXES = {
    "phone": [f"{HF_BUCKET_DUCKDB}/{IDX_PHONE}.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_BUCKET_DUCKDB}/{IDX_AADHAR}.{i}.parquet" for i in range(7)],
}

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES


def _new_conn() -> duckdb.DuckDBPyConnection:
    """Naya DuckDB connection — HfFileSystem register karke."""
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # ✅ HfFileSystem (v1.6.0+) register karo — bucket support ke saath
    fs = HfFileSystem(token=HF_TOKEN if HF_TOKEN.startswith("hf_") else None)
    con.register_filesystem(fs)

    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_parquet([{lst}], union_by_name=true)"
        )
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con


def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


# ── Dedup & Connected Records ───────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()


def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected


def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out


# ── Search Logic ────────────────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        if field == "phoneNumber" and _idx_ready("phone"):
            view = "people_phone"
        elif field == "aadharNumber" and _idx_ready("aadhar"):
            view = "people_aadhar"
        else:
            return {"field": field, "value": value, "mode": mode,
                    "count": 0, "results": []}
        sql = (
            f"SELECT * FROM {view} WHERE {field} = '{v}' "
            f"LIMIT {limit * DUPLICATE_CAP + 20}"
        )
    elif mode == "contains":
        if field == "name":
            return {"field": field, "value": value, "mode": mode,
                    "count": 0, "results": []}
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = (
            f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}%' "
            f"ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode,
            "count": len(results), "results": results}


def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8

    if not is_num:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    all_rows, searched = [], []

    # ✅ Pehle PHONE index (saari 7 files) — fast
    if _idx_ready("phone"):
        r = _run_field_search("phoneNumber", q, "exact", limit)
        all_rows.extend(r["results"])
        searched.append("phoneNumber")

    # ✅ Agar phone mein nahi mila toh AADHAR index (saari 7 files)
    if not all_rows and _idx_ready("aadhar"):
        r = _run_field_search("aadharNumber", q, "exact", limit)
        all_rows.extend(r["results"])
        searched.append("aadharNumber")

    # Enrich — connected numbers se extra records
    if all_rows:
        connected_searches = set()
        for row in all_rows[:3]:
            for nf in NUMBER_FIELDS:
                nv = row.get(nf)
                if nv and str(nv) not in connected_searches:
                    connected_searches.add(str(nv))
                    extra = _run_field_search(nf, str(nv), "exact", 2)
                    if extra["count"]:
                        all_rows.extend(extra["results"])
                        if nf not in searched:
                            searched.append(nf)

    all_rows = _cap_duplicates(all_rows)[:limit]

    return {
        "query": q, "searched_fields": searched,
        "count": len(all_rows), "results": all_rows,
    }


# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="ICMR + HITEK Search API")


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


# ── Custom HTML UI (Gradio ki jagah) ────────────────────────────────────────
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
        "bucket": HF_BUCKET_ID,
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
    }


@app.get("/debug")
def debug():
    """✅ PEHLE YE CHALAO — bucket aur token check karne ke liye."""
    try:
        con = _get_conn()
        sample = con.execute(
            f"SELECT * FROM read_parquet('{REMOTE_INDEXES['phone'][0]}', union_by_name=true) LIMIT 1"
        ).fetchdf()
        files = con.execute(
            f"SELECT COUNT(*) FROM glob('{HF_BUCKET_DUCKDB}/{IDX_PHONE}.*.parquet')"
        ).fetchone()[0]
        con.close()
        return {
            "status": "connected",
            "bucket": HF_BUCKET_ID,
            "token_loaded": bool(HF_TOKEN and HF_TOKEN.startswith("hf_")),
            "phone_files_found": files,
            "columns": list(sample.columns) if not sample.empty else [],
        }
    except Exception as e:
        return {
            "status": "error",
            "bucket": HF_BUCKET_ID,
            "token_loaded": bool(HF_TOKEN and HF_TOKEN.startswith("hf_")),
            "error": str(e),
        }


@app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True),
):
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile")

    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(
            pool, _run_field_search, field, q_val, mode, limit
        )
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)

    result = {"success": bool(data["count"]), **data, "number": q_val,
              "total": data["count"]}
    content = json.dumps(result, indent=2 if pretty else None,
                         ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")


@app.get("/info/{number}")
async def full_info(number: str, limit: int = Query(10, ge=1, le=1000),
                    pretty: bool = Query(True)):
    number = number.strip()
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _unified_search, number, limit)
    result = {
        "success": bool(data["count"]),
        "number": number,
        "number_type": (
            "phone" if len(number) in (10, 11)
            else "aadhar" if len(number) == 12
            else "unknown"
        ),
        **data,
    }
    content = json.dumps(result, indent=2 if pretty else None,
                         ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")


@app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")

    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            pool, _run_field_search,
            item.get("field", "phoneNumber"),
            item.get("value", ""),
            item.get("mode", "exact"),
            int(item.get("limit", req.limit)),
        )
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return Response(
        content=json.dumps(
            {"searches": len(req.queries), "results": list(results)},
            indent=2, ensure_ascii=False, default=str,
        ),
        media_type="application/json",
    )


# ── Pinger (keeps app alive) ──────────────────────────────────────────────
async def pinger():
    port = os.getenv("PORT", "10000")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                print(f"[Pinger] Status: {resp.status_code}")
            except Exception as e:
                print(f"[Pinger] Error: {e}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(pinger())


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"\n🚀 ICMR Search API chal raha hai: http://0.0.0.0:{port}")
    print(f"📖 Docs: http://localhost:{port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
