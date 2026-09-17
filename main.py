from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import duckdb
import os
import threading
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hitek")

app = FastAPI(title="Hitek Data Gateway", version="2.1")

# ---------- DuckDB ----------
con = duckdb.connect(database=":memory:")
try:
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET enable_http_metadata_cache=true;")
    con.execute("SET enable_object_cache=true;")
    log.info("[DuckDB] httpfs loaded successfully")
except Exception as e:
    log.error(f"[DuckDB] httpfs setup failed: {e}")

# ---------- ✅ FIXED URL — buckets, NOT datasets ----------
HF_BASE = "https://huggingface.co/buckets/CutehackX/hitek-data-bucket/resolve/main"

SHARDS = list("0123456789")
loaded_shards = set()
load_lock = threading.Lock()

# ---------- Landing Page (same as before) ----------
LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hitek Data Gateway - LIVE</title>
    <style>
        body { margin: 0; overflow: hidden; background-color: #050505; color: #00ffcc; font-family: 'Courier New', Courier, monospace; }
        #canvas-container { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; }
        .overlay { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; background: rgba(10, 10, 10, 0.85); padding: 50px; border: 1px solid #00ffcc; border-radius: 12px; box-shadow: 0 0 30px rgba(0, 255, 204, 0.3); backdrop-filter: blur(5px); }
        h1 { margin: 0 0 15px 0; font-size: 3.5em; text-transform: uppercase; letter-spacing: 6px; text-shadow: 0 0 15px #00ffcc; }
        p { font-size: 1.2em; margin: 8px 0; color: #ccc; }
        .highlight { color: #00ffcc; font-weight: bold; }
        .status-box { margin-top: 30px; font-weight: bold; padding: 15px; border-radius: 8px; background: rgba(0, 255, 204, 0.1); border: 1px solid rgba(0, 255, 204, 0.5); font-size: 1.1em; }
        .blinking { animation: blinker 1.5s linear infinite; display: inline-block; }
        @keyframes blinker { 50% { opacity: 0; } }
    </style>
</head>
<body>
    <div id="canvas-container"></div>
    <div class="overlay">
        <h1>SYSTEM ONLINE</h1>
        <p>API Gateway is <span class="highlight">Active & Secured</span></p>
        <p>Parquet Cloud Engine: <span class="highlight">Connected</span></p>
        <div class="status-box"><span class="blinking" style="color: #00ffcc;">●</span> HTTP 200 OK - LISTENING FOR QUERIES</div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('canvas-container').appendChild(renderer.domElement);
        const geometry = new THREE.BufferGeometry();
        const vertices = [];
        for (let i = 0; i < 8000; i++) {
            vertices.push(THREE.MathUtils.randFloatSpread(3000));
            vertices.push(THREE.MathUtils.randFloatSpread(3000));
            vertices.push(THREE.MathUtils.randFloatSpread(3000));
        }
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        const material = new THREE.PointsMaterial({ color: 0x00ffcc, size: 2.5, transparent: true, opacity: 0.8 });
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        camera.position.z = 1200;
        function animate() {
            requestAnimationFrame(animate);
            points.rotation.x += 0.0005;
            points.rotation.y += 0.001;
            renderer.render(scene, camera);
        }
        animate();
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
"""

# ---------- ❌ PRELOADING DISABLED ----------
# Tere files 5.4 GB each hain — 10 shards = 50+ GB. Render free tier = 512 MB.
# Preload IMPOSSIBLE hai. Direct HF read hi karna padega.
# Isliye preload function hata diya.

@app.on_event("startup")
async def startup_event():
    log.info("[Startup] Using direct HuggingFace parquet reads (no preload — files too large)")


# ---------- Exception Handler ----------
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(status_code=404, content={
            "status": "rejected",
            "message": "Invalid endpoint. STRICTLY use /FetchData?Number=XXXXXXXXXX",
            "Developer": "@Maybechx"
        })
    return JSONResponse(status_code=exc.status_code, content={
        "detail": exc.detail, "Developer": "@Maybechx"
    })


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def root_landing_page():
    return HTMLResponse(content=LANDING_PAGE_HTML, status_code=200)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "direct-read",
        "bucket": HF_BASE,
        "Developer": "@Maybechx"
    }


@app.get("/FetchData")
def fetch_data(Number: str = Query(None)):
    # ---- Validation ----
    if not Number or not Number.isdigit() or not (10 <= len(Number) <= 15):
        return JSONResponse(status_code=400, content={
            "status": "rejected",
            "message": "Invalid parameter. STRICTLY use /FetchData?Number=XXXXXXXXXX",
            "Developer": "@Maybechx"
        })

    last_digit = Number[-1]
    main_url = f"{HF_BASE}/final_master_shard_{last_digit}.parquet"
    alt_url = f"{HF_BASE}/alt_master_shard_{last_digit}.parquet"

    log.info(f"[Fetch] Number={Number} shard={last_digit}")

    main_records, alt_records = [], []

    # Main shard query
    try:
        main_records = con.execute(
            "SELECT * FROM read_parquet(?) WHERE mobile = ?",
            [main_url, Number]
        ).df().to_dict(orient="records")
        log.info(f"[Main] {len(main_records)} records found")
    except Exception as e:
        log.error(f"[Main error] {e}")

    # Alt shard query
    try:
        alt_records = con.execute(
            "SELECT * FROM read_parquet(?) WHERE alt = ?",
            [alt_url, Number]
        ).df().to_dict(orient="records")
        log.info(f"[Alt] {len(alt_records)} records found")
    except Exception as e:
        log.error(f"[Alt error] {e}")

    if not main_records and not alt_records:
        return JSONResponse(status_code=404, content={
            "status": "not_found",
            "phone": Number,
            "Developer": "@Maybechx"
        })

    return {
        "status": "success",
        "Data": {
            "Main_Records": main_records,
            "Alt_Records": alt_records
        },
        "Developer": "@Maybechx"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
