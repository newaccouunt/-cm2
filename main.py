import os
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── ✅ MASTER FIX: Hugging Face Buckets ke liye hf:// protocol ──
from huggingface_hub import HfFileSystem

# DuckDB ko HF filesystem ke baare mein batao (buckets ke liye zaroori)
duckdb.register_filesystem(HfFileSystem())

# ── Config ──
BASE = os.path.dirname(os.path.abspath(__file__))
# ✅ NAYA BUCKET PATH (datasets nahi, buckets)
HF_BUCKET_PATH = "hf://buckets/bronx-ultra/icrm-hitek-full-db-mixed-bucket"

PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

# ── DuckDB Connection Pool ──
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")

def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    
    # ✅ Buckets ke liye views banao (hf:// protocol)
    phone_urls = [f"{HF_BUCKET_PATH}/idx_phone.{i}.parquet" for i in range(7)]
    aadhar_urls = [f"{HF_BUCKET_PATH}/idx_aadhar.{i}.parquet" for i in range(7)]
    
    # Phone index view
    phone_lst = ", ".join(f"'{u}'" for u in phone_urls)
    con.execute(f"CREATE OR REPLACE VIEW people_phone AS SELECT * FROM read_parquet([{phone_lst}])")
    
    # Aadhar index view
    aadhar_lst = ", ".join(f"'{u}'" for u in aadhar_urls)
    con.execute(f"CREATE OR REPLACE VIEW people_aadhar AS SELECT * FROM read_parquet([{aadhar_lst}])")
    
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con

def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = getattr(_thread_local, "id", None)
    if ident is None:
        with _conns_lock:
            ident = len(_conns)
            _thread_local.id = ident
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]

# ── Dedup & Connected Records (same as your old code) ──
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
        if raw is None: continue
        value = str(raw).strip()
        if not value or value in seen: continue
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

# ── Search Logic (same as your old code) ──
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        if field == "phoneNumber":
            view = "people_phone"
        elif field == "aadharNumber":
            view = "people_aadhar"
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    elif mode == "contains":
        if field == "name":
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}

def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8
    if is_num:
        all_rows = []
        searched = []
        r = _run_field_search("phoneNumber", q, "exact", limit)
        all_rows.extend(r["results"]); searched.append("phoneNumber")
        if not all_rows:
            r = _run_field_search("aadharNumber", q, "exact", limit)
            all_rows.extend(r["results"]); searched.append("aadharNumber")
        all_rows = _cap_duplicates(all_rows)[:limit]
        return {"query": q, "searched_fields": searched, "count": len(all_rows), "results": all_rows}
    else:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

# ── FastAPI + Gradio (same as your old code, no changes needed) ──
fastapi_app = FastAPI(title="ICMR + HITEK Search API")

class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10

@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "records": 2_504_793_870,
        "indexes": {"phone": True, "aadhar": True},
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@kzr0x | channel @api_wallah",
    }

@fastapi_app.get("/health")
def health():
    return {"status": "ok", "indexes": {"phone": True, "aadhar": True}}

@fastapi_app.get("/search")
async def search(q: str | None = Query(None), mobile: str | None = Query(None), field: str | None = Query(None), mode: str = Query("exact"), limit: int = Query(10, ge=1, le=1000), pretty: bool = Query(True)):
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile")
    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)
    result = {"success": bool(data["count"]), **data, "number": q_val, "total": data["count"]}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(pool, _run_field_search, item.get("field", "phoneNumber"), item.get("value", ""), item.get("mode", "exact"), int(item.get("limit", req.limit))) for item in req.queries]
    results = await asyncio.gather(*tasks)
    return Response(content=json.dumps({"searches": len(req.queries), "results": list(results)}, indent=2, ensure_ascii=False), media_type="application/json")

# ── Pinger ──
async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                print(f"[Pinger] {resp.status_code}")
            except Exception as e:
                print(f"[Pinger] Error: {e}")

@fastapi_app.on_event("startup")
async def startup_event():
    asyncio.create_task(pinger())

# ── Gradio UI (same as your old code) ──
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val:
            lines.append(f"**{field}:** {val}")
    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")
    return "\n\n".join(lines)

def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Kuch toh search karo — phone, aadhar, ya name daalo."
    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    count = data["count"]
    results = data["results"]
    searched = ", ".join(data.get("searched_fields", []))
    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ **No data found** for this number."
    header = f"🔍 **Query:** `{q}`  |  **Found:** {count} results  |  **Searched:** {searched}\n\n---\n\n"
    parts = [f"### Result {i}\n{format_result(row)}" for i, row in enumerate(results, 1)]
    return header + "\n\n---\n\n".join(parts)

def build_ui():
    with gr.Blocks(title="ICMR Search API", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API")
        gr.Markdown("Search **2.5 billion records** — phone, Aadhaar, name, address & more")
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Search Query", placeholder="Phone number, Aadhaar, ya name daalo...", lines=1)
            with gr.Column(scale=1):
                limit_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Max Results")
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        search_btn.click(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        query_input.submit(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        gr.Markdown("---\n👨‍💻 **Developer:** @kzr0x  |  📢 **Channel:** @api_wallah")
    return demo

demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
