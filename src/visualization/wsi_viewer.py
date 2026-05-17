"""
Browser-based whole-slide viewer.

Serves SVS files as Deep Zoom tiles via Flask + OpenSlide, with optional
predictions.geojson overlay rendered by OpenSeadragon. Tiles stream on
demand — nothing is downloaded locally.

Usage on SCC (from project root):

    qrsh -P rise2019 -l h_rt=2:00:00 -pe omp 2
    hostname -s                                    # note the compute node
    source vascuenv/bin/activate
    python -m src.visualization.wsi_viewer \
        --slide-dir "/projectnb/rise2019/JC_CTE_Images/AI export/Frontal Cortex" \
        --pred-dir  src/outputs \
        --port 5000

Then from your laptop:

    ssh -L 5000:<compute-node>:5000 <user>@scc1.bu.edu

Open http://localhost:5000 in the browser.
"""

import argparse
from io import BytesIO
from pathlib import Path

import openslide
from flask import Flask, abort, jsonify, render_template_string, send_file
from openslide.deepzoom import DeepZoomGenerator

app = Flask(__name__)

# populated in __main__
SLIDE_DIR: Path = None
PRED_DIR: Path = None
TILE_SIZE = 256
slides_cache: dict = {}


def get_dz(name: str):
    """Return (OpenSlide, DeepZoomGenerator) for the named slide, opening lazily."""
    if name not in slides_cache:
        path = SLIDE_DIR / f"{name}.svs"
        if not path.exists():
            abort(404, f"slide not found: {path}")
        osr = openslide.OpenSlide(str(path))
        dz = DeepZoomGenerator(osr, tile_size=TILE_SIZE, overlap=0, limit_bounds=True)
        slides_cache[name] = (osr, dz)
    return slides_cache[name]


@app.route("/")
def index():
    slides = sorted(p.stem for p in SLIDE_DIR.glob("*.svs"))
    return render_template_string(INDEX_HTML, slides=slides)


@app.route("/<name>/")
def viewer(name):
    return render_template_string(VIEWER_HTML, name=name)


@app.route("/<name>.dzi")
def dzi(name):
    _, dz = get_dz(name)
    return dz.get_dzi("jpeg"), 200, {"Content-Type": "application/xml"}


@app.route("/<name>_files/<int:level>/<int:col>_<int:row>.jpeg")
def tile(name, level, col, row):
    _, dz = get_dz(name)
    try:
        img = dz.get_tile(level, (col, row))
    except (ValueError, KeyError):
        abort(404)
    buf = BytesIO()
    img.save(buf, "jpeg", quality=80)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/<name>/predictions.geojson")
def predictions(name):
    p = (PRED_DIR / name / "predictions.geojson").resolve()
    if not p.exists():
        return jsonify({"type": "FeatureCollection", "features": []})
    return send_file(str(p), mimetype="application/json")


INDEX_HTML = """
<!doctype html>
<title>VascuPath slides</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 2rem auto; }
  ul   { line-height: 1.8; }
  a    { text-decoration: none; }
</style>
<h1>Slides ({{slides|length}})</h1>
<ul>
{% for s in slides %}
  <li><a href="/{{s}}/">{{s}}</a></li>
{% else %}
  <li><em>no slides found</em></li>
{% endfor %}
</ul>
"""


VIEWER_HTML = """
<!doctype html>
<title>{{name}}</title>
<script src="https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/openseadragon.min.js"></script>
<style>
  body { margin: 0; font-family: system-ui, sans-serif; }
  #osd { width: 100vw; height: 100vh; background: #111; }
  #hud {
    position: fixed; top: 10px; left: 10px; padding: 6px 10px;
    background: rgba(0,0,0,0.6); color: #fff; font-size: 12px; border-radius: 4px;
  }
  label { margin-right: 12px; cursor: pointer; }
</style>
<div id="osd"></div>
<div id="hud">
  <strong>{{name}}</strong> &nbsp;
  <label><input type="checkbox" id="toggle-overlays" checked> overlays</label>
  <a href="/" style="color:#9cf">← all slides</a>
</div>
<script>
const SLIDE = "{{name}}";

const v = OpenSeadragon({
  id: "osd",
  prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/images/",
  tileSources: "/" + SLIDE + ".dzi",
  showNavigator: true,
  navigatorPosition: "BOTTOM_RIGHT",
});

v.addHandler("open", async () => {
  const gj  = await (await fetch("/" + SLIDE + "/predictions.geojson")).json();
  const img = v.world.getItemAt(0);

  for (const f of gj.features) {
    const ring = f.geometry.coordinates[0];
    const xs = ring.map(c => c[0]), ys = ring.map(c => c[1]);
    const x = Math.min(...xs), y = Math.min(...ys);
    const w = Math.max(...xs) - x, h = Math.max(...ys) - y;
    const [r,g,b] = f.properties.classification.color;
    const cls    = f.properties.classification.name;

    const div = document.createElement("div");
    div.className = "overlay";
    div.title = cls;
    div.style.background = `rgba(${r},${g},${b},0.35)`;
    div.style.border     = `1px solid rgba(${r},${g},${b},0.9)`;
    v.addOverlay({ element: div, location: img.imageToViewportRectangle(x, y, w, h) });
  }
});

document.getElementById("toggle-overlays").addEventListener("change", e => {
  for (const o of document.querySelectorAll(".overlay")) {
    o.style.display = e.target.checked ? "" : "none";
  }
});
</script>
"""


def main():
    global SLIDE_DIR, PRED_DIR
    ap = argparse.ArgumentParser(description="Browser-based WSI viewer")
    ap.add_argument("--slide-dir", required=True, type=Path,
                    help="Directory containing .svs files")
    ap.add_argument("--pred-dir",  required=True, type=Path,
                    help="Directory containing per-slide subfolders with predictions.geojson")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    SLIDE_DIR = args.slide_dir.resolve()
    PRED_DIR  = args.pred_dir.resolve()

    print(f"Slide dir : {SLIDE_DIR}")
    print(f"Pred dir  : {PRED_DIR}")
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
