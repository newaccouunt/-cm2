import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from typing import Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel
import pyarrow.parquet as pq
import pandas as pd
from io import BytesIO

# ── Config ──────────────────────────────────────────────────────────────────
HF_DATASET_URL = os.environ.get(
    "ICMR_HF_DATASET_URL",
    "https://huggingface.co/datasets/rehuuuu/icrm-hitek-fulldb/resolve/main",
).rstrip("/")

HF_REPO_ID = "rehuuuu/icrm-hitek-fulldb"

PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "4"))
TIMEOUT = int(os.environ.get("ICMR_TIMEOUT", "90"))
DUPLICATE_CAP = 2
MAX_CACHED_DFS = int(os.environ.get("ICMR_MAX_CACHED_DFS", "2"))

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

COLUMN_ALIASES = {
    "phoneNumber": ["phoneNumber", "phone_number", "phone", "mobile",
                    "mobileNumber", "mobile_number", "contact", "contactNumber"],
    "aadharNumber": ["aadharNumber", "aadhar_number", "aadhaarNumber",
                     "aadhaar_number", "aadhar", "aadhaar", "uid"],
    "otherNumber": ["otherNumber", "other_number", "altNumber",
                    "alternateNumber", "other"],
    "name": ["name", "fullName", "full_name", "Name"],
    "fathersName": ["fathersName", "fathers_name", "fatherName", "father_name"],
    "address": ["address", "Address", "addr"],
    "district": ["district", "District", "dist"],
    "pincode": ["pincode", "pinCode", "pin_code", "zip", "zipcode"],
    "state": ["state", "State"],
    "town": ["town", "Town", "city", "City"],
    "source": ["source", "Source", "src"],
}

# ── Caches & Locks ──────────────────────────────────────────────────────────
_parquet_cache: dict[str, list[str]] = {}
_parquet_lock = threading.Lock()
_df_cache: dict[str, pd.DataFrame] = {}
_df_cache_lock = threading.Lock()

# ── Auto-discover parquet files ─────────────────────────────────────────────
async def discover_parquet_files(client: httpx.AsyncClient) -> dict[str, list[str]]:
    global _parquet_cache
    with _parquet_lock:
        if _parquet_cache and any(_parquet_cache.values()):
            return _parquet_cache

    found = {"phone": [], "aadhar": [], "other": []}

    # Method 1: HF Tree API
    try:
        for tree_url in [
            f"https://huggingface.co/api/datasets/{HF_REPO_ID}/tree/main",
            f"https://huggingface.co/api/datasets/{HF_REPO_ID}/tree/main?recursive=true",
        ]:
            try:
                r = await client.get(tree_url, timeout=30)
                if r.status_code == 200:
                    for item in r.json():
                        fname = item.get("path", "") or item.get("rfilename", "")
                        if fname.endswith(".parquet"):
                            url = f"{HF_DATASET_URL}/{fname}"
                            low = fname.lower()
                            if "phone" in low or "mobile" in low or "contact" in low:
                                found["phone"].append(url)
                            elif "aadhar" in low or "aadhaar" in low:
                                found["aadhar"].append(url)
                            else:
                                found["other"].append(url)
                            print(f"✅ Tree discovered: {fname}")
                    if any(found.values()):
                        break
            except Exception as e:
                print(f"⚠️ Tree API failed: {e}")
    except Exception as e:
        print(f"⚠️ Tree API outer failed: {e}")

    # Method 2: HF Dataset Info API
    if not any(found.values()):
        try:
            r = await client.get(
                f"https://huggingface.co/api/datasets/{HF_REPO_ID}", timeout=30
            )
            if r.status_code == 200:
                for s in r.json().get("siblings", []):
                    fname = s.get("rfilename", "")
                    if fname.endswith(".parquet"):
                        url = f"{HF_DATASET_URL}/{fname}"
                        low = fname.lower()
                        if "phone" in low or "mobile" in low:
                            found["phone"].append(url)
                        elif "aadhar" in low or "aadhaar" in low:
                            found["aadhar"].append(url)
                        else:
                            found["other"].append(url)
                        print(f"✅ Info API discovered: {fname}")
        except Exception as e:
            print(f"⚠️ Info API failed: {e}")

    # Method 3: Range-GET probe
    if not any(found.values()):
        print("🔎 Range-GET probe fallback...")
        patterns = [
            ("idx_phone", "phone"), ("idx_aadhar", "aadhar"),
            ("phone", "phone"), ("aadhar", "aadhar"),
            ("aadhaar", "aadhar"), ("mobile", "phone"),
            ("data", "other"), ("dataset", "other"),
            ("part", "other"), ("train", "other"),
        ]
        for prefix, key in patterns:
            for i in range(50):
                candidates = [
                    f"{prefix}.{i}.parquet",
                    f"{prefix}_{i}.parquet",
                    f"{prefix}-{i}.parquet",
                    f"{prefix}{i}.parquet",
                ]
                if i == 0:
                    candidates.insert(0, f"{prefix}.parquet")
                for fmt in candidates:
                    url = f"{HF_DATASET_URL}/{fmt}"
                    try:
                        rr = await client.get(
                            url, headers={"Range": "bytes=0-0"},
                            follow_redirects=True, timeout=10
                        )
                        if rr.status_code in (200, 206):
                            found[key].append(url)
                            print(f"✅ Probe found: {fmt}")
                    except Exception:
                        pass

    if any(found.values()):
        with _parquet_lock:
            _parquet_cache = found

    print(f"📦 Discovered: phone={len(found['phone'])}, "
          f"aadhar={len(found['aadhar'])}, other={len(found['other'])}")
    return found

# ── Download parquet with LRU cache ─────────────────────────────────────────
async def download_parquet(url: str, client: httpx.AsyncClient) -> pd.DataFrame:
    with _df_cache_lock:
        if url in _df_cache:
            return _df_cache[url]

    try:
        response = await client.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        buffer = BytesIO(response.content)
        table = pq.read_table(buffer)
        df = table.to_pandas()

        with _df_cache_lock:
            if len(_df_cache) >= MAX_CACHED_DFS:
                oldest = next(iter(_df_cache))
                _df_cache.pop(oldest, None)
                print(f"🗑️ Evicted cache: {oldest.split('/')[-1]}")
            _df_cache[url] = df

        print(f"✅ Loaded {url.split('/')[-1]} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return pd.DataFrame()

# ── Column detection ────────────────────────────────────────────────────────
def detect_column(df: pd.DataFrame, field: str) -> Optional[str]:
    if field in df.columns:
        return field
    aliases = COLUMN_ALIASES.get(field, [field])
    cols_lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    for c in df.columns:
        for alias in aliases:
            if alias.lower() in c.lower():
                return c
    return None

# ── Numeric-safe matching ───────────────────────────────────────────────────
def _normalize(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    return s

def _match(series: pd.Series, value: str) -> pd.Series:
    v = str(value).strip()
    s = _normalize(series)
    mask = s == v
    v_nz = v.lstrip("0") or v
    mask |= s.str.lstrip("0") == v_nz
    return mask

# ── Search parquet ──────────────────────────────────────────────────────────
async def search_in_parquet(field: str, value: str, limit: int = 10) -> list:
    results = []
    seen_urls = set()

    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        files_map = await discover_parquet_files(client)

        if field == "aadharNumber":
            files = files_map["aadhar"] + files_map["other"] + files_map["phone"]
        elif field in ("phoneNumber", "otherNumber"):
            files = files_map["phone"] + files_map["other"] + files_map["aadhar"]
        else:
            files = files_map["phone"] + files_map["aadhar"] + files_map["other"]

        files = [u for u in files if not (u in seen_urls or seen_urls.add(u))]

        if not files:
            print(f"⚠️ No parquet files discovered for field={field}!")
            return []

        print(f"🔍 Searching field={field} value={value} across {len(files)} files")

        async def search_one(url):
            df = await download_parquet(url, client)
            if df is None or df.empty:
                return []
            col = detect_column(df, field)
            if col is None:
                print(f"⚠️ Column '{field}' not in {url.split('/')[-1]}. "
                      f"Available: {list(df.columns)}")
                return []
            mask = _match(df[col], value)
            hits = int(mask.sum())
            if hits:
                print(f"✅ {url.split('/')[-1]}: {hits} hits in col '{col}'")
            rows = df[mask].to_dict("records")
            for r in rows:
                for std_field, aliases in COLUMN_ALIASES.items():
                    if std_field not in r:
                        for a in aliases:
                            if a in r and a != std_field:
                                r[std_field] = r[a]
                                break
            return rows

        tasks = [search_one(u) for u in files]
        for coro in asyncio.as_completed(tasks):
            try:
                rows = await coro
                for r in rows:
                    results.append(r)
                    if len(results) >= limit * 3:
                        return results
            except Exception as e:
                print(f"⚠️ Search error: {e}")

    return results

# ── Dedup + connected numbers ───────────────────────────────────────────────
def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        value = re.sub(r"\.0+$", "", str(raw).strip())
        if not value or value in seen or value.lower() == "nan":
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected

def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen, out = {}, []
    for r in rows:
        ph = str(r.get("phoneNumber", "")).strip()
        ad = str(r.get("aadharNumber", "")).strip()
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

# ── Sync wrappers ───────────────────────────────────────────────────────────
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

def _unified_search_sync(q: str, limit: int = 10) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8
    if not is_num:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    all_rows, searched = [], []

    if 8 <= len(q) <= 13:
        try:
            results = _run_async(search_in_parquet("phoneNumber", q, limit))
            if results:
                all_rows.extend(results)
                searched.append("phoneNumber")
        except Exception as e:
            print(f"Phone search error: {e}")

    if len(q) == 12:
        try:
            results = _run_async(search_in_parquet("aadharNumber", q, limit))
            if results:
                all_rows.extend(results)
                searched.append("aadharNumber")
        except Exception as e:
            print(f"Aadhar search error: {e}")

    if not all_rows:
        try:
            results = _run_async(search_in_parquet("otherNumber", q, limit))
            if results:
                all_rows.extend(results)
                searched.append("otherNumber")
        except Exception as e:
            print(f"otherNumber search error: {e}")

    all_rows = _cap_duplicates(all_rows)[:limit]
    return {
        "query": q,
        "searched_fields": searched,
        "count": len(all_rows),
        "results": all_rows,
    }

def _run_field_search_sync(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        return {"field": field, "value": value, "mode": mode, "count": 0,
                "results": [], "error": "Unknown field"}
    try:
        results = _run_async(search_in_parquet(field, value, limit))
        results = _cap_duplicates(results)[:limit]
        return {"field": field, "value": value, "mode": mode,
                "count": len(results), "results": results}
    except Exception as e:
        return {"field": field, "value": value, "mode": mode,
                "count": 0, "results": [], "error": str(e)}

# ── Self-pinger (Render free tier ke liye helpful) ──────────────────────────
async def pinger():
    render_url = os.getenv("RENDER_EXTERNAL_URL")
    port = os.getenv("PORT", "10000")
    url = f"{render_url}/health" if render_url else f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(600)  # har 10 min
            try:
                resp = await client.get(url)
                print(f"[Pinger] {url} -> {resp.status_code}")
            except Exception as e:
                print(f"[Pinger] Error: {e}")

# ── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 ICMR Search API started on Render!")
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            files = await discover_parquet_files(client)
            print(f"📋 Startup discovery: {sum(len(v) for v in files.values())} files")
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
        "dataset": HF_REPO_ID,
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@kzr0x | channel @api_wallah",
    }

@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": HF_REPO_ID,
        "discovered_files": {k: len(v) for k, v in _parquet_cache.items()},
        "cached_dfs": len(_df_cache),
        "has_data": any(_parquet_cache.values()) if _parquet_cache else False,
    }

@fastapi_app.get("/debug/files")
async def debug_files():
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        files = await discover_parquet_files(client)
    return {
        "total": sum(len(v) for v in files.values()),
        "phone": files.get("phone", []),
        "aadhar": files.get("aadhar", []),
        "other": files.get("other", []),
    }

@fastapi_app.get("/debug/columns")
async def debug_columns():
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        files = await discover_parquet_files(client)
        all_files = files.get("phone", []) + files.get("aadhar", []) + files.get("other", [])
        if not all_files:
            return {"error": "No files discovered"}
        url = all_files[0]
        df = await download_parquet(url, client)
        return {
            "file": url.split("/")[-1],
            "rows": len(df),
            "columns": list(df.columns),
            "sample": df.head(2).to_dict("records") if not df.empty else [],
        }

@fastapi_app.get("/debug/reset")
async def debug_reset():
    global _parquet_cache
    with _parquet_lock:
        _parquet_cache = {}
    with _df_cache_lock:
        _df_cache.clear()
    return {"status": "caches cleared"}

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

# ── Thread pool ─────────────────────────────────────────────────────────────
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="search")

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
    searched = ", ".join(data.get("searched_fields", [])) or "none"
    if not results:
        hint = ""
        if not (_parquet_cache and any(_parquet_cache.values())):
            hint = "\n\n🚨 **Parquet files discover nahi hui!** `/debug/files` check karo."
        return (f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n"
                f"❌ **No data found** for this number.{hint}")
    header = f"🔍 **Query:** `{q}`  |  **Found:** {count} results  |  **Searched:** {searched}\n\n---\n\n"
    parts = [f"### Result {i}\n{format_result(row)}" for i, row in enumerate(results, 1)]
    return header + "\n\n---\n\n".join(parts)

def build_ui():
    with gr.Blocks(title="ICMR Search API", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API")
        gr.Markdown(f"Search **{HF_REPO_ID}** — phone, Aadhaar & more")
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Phone number ya Aadhaar daalo...",
                    lines=1,
                )
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
- `GET /debug/files` — See discovered parquet files
- `GET /debug/columns` — See columns of first file
- `GET /debug/reset` — Clear caches & re-discover
- `GET /docs` — Swagger UI
            """)
        gr.Markdown("---\n<div style='text-align:center;color:#888;'>👨‍💻 **Developer:** @kzr0x | 📢 **Channel:** @api_wallah</div>")
    return demo

demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
