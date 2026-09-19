import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
HF_DATASET = os.environ.get("ICMR_HF_DATASET", "rehuuuu/icrm-hitek-fulldb")
HF_CONFIG = os.environ.get("ICMR_HF_CONFIG", "default")
HF_SPLIT = os.environ.get("ICMR_HF_SPLIT", "train")
HF_API_BASE = "https://datasets-server.huggingface.co"

PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
TIMEOUT = int(os.environ.get("ICMR_TIMEOUT", "120"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

# ── Thread Pool ─────────────────────────────────────────────────────────────
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="search")

# ── HF Dataset Viewer API Search ────────────────────────────────────────────
async def _hf_search(field: str, value: str, limit: int = 10) -> list[dict]:
    """
    Use Hugging Face Dataset Viewer API to search server-side.
    No file download needed. Works even for 227GB datasets.
    """
    url = f"{HF_API_BASE}/filter"
    params = {
        "dataset": HF_DATASET,
        "config": HF_CONFIG,
        "split": HF_SPLIT,
        "where": f"\"{field}\"='{value}'",
        "limit": str(limit),
        "offset": "0",
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                data = r.json()
                rows = data.get("rows", [])
                # HF returns list of {"row": {...}, "row_idx": n}
                out = []
                for item in rows:
                    row = item.get("row", {}) if isinstance(item, dict) else {}
                    if row:
                        out.append(row)
                print(f"✅ HF API: {field}='{value}' -> {len(out)} rows")
                return out
            else:
                print(f"⚠️ HF API error {r.status_code}: {r.text[:200]}")
                return []
    except Exception as e:
        print(f"❌ HF API exception: {e}")
        return []

async def _hf_search_multi(fields: list[str], value: str, limit: int = 10) -> list[dict]:
    """Search same value across multiple fields in parallel."""
    tasks = [_hf_search(f, value, limit) for f in fields]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
    return out

# ── Dedup & Connected Records ───────────────────────────────────────────────
def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        value = str(raw).strip()
        if not value or value in seen or value.lower() == "nan":
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected

def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen = {}
    out = []
    for r in rows:
        ph = str(r.get("phoneNumber", "") or "").strip()
        ad = str(r.get("aadharNumber", "") or "").strip()
        key = (ph, ad) if (ph or ad) else (str(r.get("name", "")), str(r.get("fathersName", "")))
        n = seen.get(key, 0)
        if n < DUPLICATE_CAP:
            seen[key] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out

# ── Sync wrapper ────────────────────────────────────────────────────────────
def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as ex:
                return ex.submit(lambda: asyncio.run(coro)).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

# ── Search Logic ────────────────────────────────────────────────────────────
def _unified_search_sync(q: str, limit: int = 10) -> dict:
    q = q.strip()
    if not q:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    is_num = q.isdigit() and len(q) >= 8
    all_rows = []
    searched = []

    if is_num:
        # Try phone
        if 8 <= len(q) <= 13:
            rows = _run_async(_hf_search("phoneNumber", q, limit))
            if rows:
                all_rows.extend(rows); searched.append("phoneNumber")
        # Try aadhar
        if len(q) == 12:
            rows = _run_async(_hf_search("aadharNumber", q, limit))
            if rows:
                all_rows.extend(rows); searched.append("aadharNumber")
        # Fallback: otherNumber
        if not all_rows:
            rows = _run_async(_hf_search("otherNumber", q, limit))
            if rows:
                all_rows.extend(rows); searched.append("otherNumber")
    else:
        # Text search on name
        rows = _run_async(_hf_search("name", q, limit))
        if rows:
            all_rows.extend(rows); searched.append("name")

    all_rows = _cap_duplicates(all_rows)[:limit]
    return {
        "query": q,
        "searched_fields": searched,
        "count": len(all_rows),
        "results": all_rows,
    }

def _run_field_search_sync(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": [], "error": "Unknown field"}
    try:
        rows = _run_async(_hf_search(field, value, limit))
        rows = _cap_duplicates(rows)[:limit]
        return {"field": field, "value": value, "mode": mode, "count": len(rows), "results": rows}
    except Exception as e:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": [], "error": str(e)}

# ── Pinger ──────────────────────────────────────────────────────────────────
async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                print(f"[Pinger] Status: {resp.status_code}")
            except Exception as e:
                print(f"[Pinger] Error: {e}")

# ── FastAPI Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 ICMR Search API started!")
    # Verify HF dataset is reachable
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.get(f"{HF_API_BASE}/splits", params={"dataset": HF_DATASET})
            if r.status_code == 200:
                print(f"✅ HF dataset reachable: {HF_DATASET}")
            else:
                print(f"⚠️ HF /splits returned {r.status_code}")
    except Exception as e:
        print(f"Pre-warm failed: {e}")
    asyncio.create_task(pinger())
    yield
    print("👋 Shutting down...")

# ── FastAPI App ─────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="ICMR + HITEK Search API", lifespan=lifespan)

class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10

@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "dataset": HF_DATASET,
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@kzr0x | channel @api_wallah",
    }

@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": HF_DATASET,
        "method": "huggingface-dataset-viewer-api",
    }

@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    aadhar: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True),
):
    if aadhar:
        q_val = aadhar.strip(); field = "aadharNumber"
    elif mobile:
        q_val = mobile.strip(); field = "phoneNumber"
    elif q:
        q_val = q.strip()
    else:
        raise HTTPException(422, "Provide q, mobile, or aadhar")

    if not q_val:
        raise HTTPException(422, "Query cannot be empty")

    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(pool, _run_field_search_sync, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search_sync, q_val, limit)

    result = {"success": bool(data.get("count", 0) > 0), **data, "number": q_val}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")

@fastapi_app.get("/search/phone/{number}")
async def search_phone(number: str, limit: int = Query(10, ge=1, le=100), pretty: bool = Query(True)):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _run_field_search_sync, "phoneNumber", number, "exact", limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": number}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")

@fastapi_app.get("/search/aadhar/{number}")
async def search_aadhar(number: str, limit: int = Query(10, ge=1, le=100), pretty: bool = Query(True)):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _run_field_search_sync, "aadharNumber", number, "exact", limit)
    result = {"success": bool(data.get("count", 0) > 0), **data, "number": number}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")

@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 20:
        raise HTTPException(400, "max 20 queries per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(pool, _run_field_search_sync,
                             item.get("field", "phoneNumber"),
                             item.get("value", ""),
                             item.get("mode", "exact"),
                             int(item.get("limit", req.limit)))
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return {"searches": len(req.queries), "results": list(results)}

# ── Gradio UI ───────────────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val and str(val) != "nan":
            lines.append(f"**{field}:** {val}")
    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")
    return "\n\n".join(lines)

def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Kuch toh search karo — phone ya aadhar number daalo."
    q = query.strip()
    data = _unified_search_sync(q, int(limit))
    count = data.get("count", 0)
    results = data.get("results", [])
    searched = ", ".join(data.get("searched_fields", []))
    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched or 'none'}\n\n❌ **No data found** for this number."
    header = f"🔍 **Query:** `{q}`  |  **Found:** {count} results  |  **Searched:** {searched}\n\n---\n\n"
    parts = [f"### Result {i}\n{format_result(row)}" for i, row in enumerate(results, 1)]
    return header + "\n\n---\n\n".join(parts)

def build_ui():
    with gr.Blocks(title="ICMR Search API", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API")
        gr.Markdown("Search **rehuuuu/icrm-hitek-fulldb** — phone, Aadhaar & more")
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Search Query", placeholder="Phone number ya Aadhaar daalo...", lines=1)
            with gr.Column(scale=1):
                limit_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Max Results")
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        search_btn.click(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        query_input.submit(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        gr.Markdown("---")
        with gr.Accordion("📡 API Info", open=False):
            gr.Markdown("""
**Endpoints:**
- `GET /search?q=<number>` — Auto-detect search
- `GET /search/phone/<number>` — Phone search
- `GET /search/aadhar/<number>` — Aadhar search
- `GET /health` — Health check
- `GET /docs` — Swagger UI

**Note:** Uses Hugging Face Dataset Viewer API — no 227GB download, results aate hain server-side se.
            """)
        gr.Markdown("---\n<div style='text-align:center;color:#888;'>👨‍💻 **Developer:** @kzr0x | 📢 **Channel:** @api_wallah</div>")
    return demo

demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
