import asyncio
import json
import os
from typing import Any

import httpx
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")

HF_BUCKET_REPO = os.environ.get(
    "HF_BUCKET_REPO",
    "bronx-ultra/icrm-hitek-full-db-mixed-bucket"
)

# Bucket ke index files
PHONE_FILES = [f"idx_phone.{i}.parquet" for i in range(7)]
AADHAR_FILES = [f"idx_aadhar.{i}.parquet" for i in range(7)]

DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]


# ── Parquet Download + Search ───────────────────────────────────────────────
def _search_in_file(filename: str, field: str, value: str, limit: int = 10) -> list[dict]:
    """Ek parquet file me search karo."""
    try:
        # HuggingFace se file download karo
        local_path = hf_hub_download(
            repo_id=HF_BUCKET_REPO,
            filename=filename,
            repo_type="bucket",
            token=HF_TOKEN if HF_TOKEN else None,
        )
        
        # Parquet padho
        table = pq.read_table(local_path, columns=SEARCH_FIELDS)
        df = table.to_pandas()
        
        # Search karo
        if field in df.columns:
            mask = df[field].astype(str).str.strip() == str(value).strip()
            matches = df[mask]
            if not matches.empty:
                # NaN clean karo
                results = []
                for _, row in matches.iterrows():
                    record = {}
                    for k, v in row.items():
                        if pd.isna(v):
                            record[k] = None
                        else:
                            record[k] = v
                    results.append(record)
                    if len(results) >= limit:
                        break
                return results
        return []
    except Exception as e:
        print(f"⚠️ Error in {filename}: {e}")
        return []


def _search_field(field: str, value: str, limit: int = 10) -> list[dict]:
    """Saari files me search karo — parallel."""
    if field in ("phoneNumber", "otherNumber"):
        files = PHONE_FILES
    elif field == "aadharNumber":
        files = AADHAR_FILES
    else:
        files = PHONE_FILES
    
    all_results = []
    for filename in files:
        results = _search_in_file(filename, field, value, limit)
        all_results.extend(results)
        if len(all_results) >= limit:
            break
    
    return all_results


# ── Dedup ───────────────────────────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = str(row.get("phoneNumber") or "").strip()
    ad = str(row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (str(row.get("name") or ""), str(row.get("fathersName") or ""))


def _connected(row: dict) -> list[dict]:
    out, seen = [], set()
    for f in NUMBER_FIELDS:
        v = row.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if not s or s == "nan" or s in seen:
            continue
        seen.add(s)
        out.append({"field": f, "value": s})
    return out


def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen, out = {}, []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            rec = dict(r)
            rec["connected_numbers"] = _connected(rec)
            out.append(rec)
    return out


# ── Unified Search ──────────────────────────────────────────────────────────
def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    if not q or not q.isdigit() or len(q) < 8:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}
    
    all_rows, searched = [], []
    
    # Phone pehle
    if len(q) in (10, 11):
        rows = _search_field("phoneNumber", q, limit)
        if rows:
            all_rows.extend(rows)
            searched.append("phoneNumber")
    
    # Aadhar
    if len(q) == 12:
        rows = _search_field("aadharNumber", q, limit)
        if rows:
            all_rows.extend(rows)
            searched.append("aadharNumber")
    
    # Enrich
    if all_rows:
        seen_nums = set()
        for row in all_rows[:2]:
            for nf in NUMBER_FIELDS:
                nv = row.get(nf)
                if nv and str(nv) not in seen_nums:
                    seen_nums.add(str(nv))
                    extra = _search_field(nf, str(nv), 2)
                    if extra:
                        all_rows.extend(extra)
                        if nf not in searched:
                            searched.append(nf)
    
    all_rows = _cap_duplicates(all_rows)[:limit]
    
    return {
        "query": q, "searched_fields": searched,
        "count": len(all_rows), "results": all_rows,
    }


# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(title="ICMR Search API")


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


# ── UI ──────────────────────────────────────────────────────────────────────
UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICMR Search</title>
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
  <h1>🔍 ICMR Search</h1>
  <div class="sub">Phone ya Aadhaar daalo</div>
  <div class="row">
    <input id="q" placeholder="10-digit phone ya 12-digit aadhar..." autofocus>
    <button id="btn" onclick="go()">Search</button>
  </div>
  <div id="out"></div>
  <div class="footer">👨‍💻 <a href="https://t.me/api_wallah">@kzr0x</a> | 📢 <a href="https://t.me/api_wallah">@api_wallah</a></div>
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
    return UI


@app.get("/health")
def health():
    return {
        "status": "ok",
        "bucket": HF_BUCKET_REPO,
        "token_loaded": bool(HF_TOKEN),
    }


@app.get("/debug")
def debug():
    """PEHLE YE CHALAO."""
    try:
        # Ek file try karo
        path = hf_hub_download(
            repo_id=HF_BUCKET_REPO,
            filename=PHONE_FILES[0],
            repo_type="bucket",
            token=HF_TOKEN if HF_TOKEN else None,
        )
        # Columns check karo
        schema = pq.read_schema(path)
        return {
            "status": "connected",
            "bucket": HF_BUCKET_REPO,
            "token_loaded": bool(HF_TOKEN),
            "file": PHONE_FILES[0],
            "columns": schema.names,
        }
    except Exception as e:
        return {
            "status": "error",
            "bucket": HF_BUCKET_REPO,
            "token_loaded": bool(HF_TOKEN),
            "error": str(e),
        }


@app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True),
):
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile")
    
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _unified_search, q_val, limit)
    result = {"success": bool(data["count"]), **data, "number": q_val}
    content = json.dumps(result, indent=2 if pretty else None,
                         ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")


@app.get("/info/{number}")
async def full_info(
    number: str,
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True),
):
    number = number.strip()
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _unified_search, number, limit)
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


# ── Pinger ──────────────────────────────────────────────────────────────────
async def pinger():
    port = os.getenv("PORT", "10000")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                print(f"[Pinger] {resp.status_code}")
            except Exception as e:
                print(f"[Pinger] {e}")


@app.on_event("startup")
async def startup():
    asyncio.create_task(pinger())


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"\n🚀 ICMR Search API: http://0.0.0.0:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
