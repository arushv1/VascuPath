"""
Browser-based whole-slide viewer.

Serves SVS files as Deep Zoom tiles via Flask + OpenSlide, with optional
overlays rendered by OpenSeadragon:
  * vessel predictions from <pred-dir>/<slide>/predictions.geojson
  * ABMIL attention heatmaps (one per comparison) computed on demand from
    <features-dir>/<slide>.pt + <mil-dir>/mil_<comparison>.pth

Tiles stream on demand — nothing is downloaded locally.

Usage on SCC (from project root):

    qrsh -P rise2019 -l h_rt=2:00:00 -pe omp 2
    hostname -s                                    # note the compute node
    source vascuenv/bin/activate
    python -m src.visualization.wsi_viewer \
        --slide-dir "/projectnb/rise2019/JC_CTE_Images/AI export/Frontal Cortex" \
        --pred-dir  src/outputs \
        --features-dir data/processed \
        --mil-dir mil \
        --port 5000

Then from your laptop:

    ssh -L 5000:<compute-node>:5000 <user>@scc1.bu.edu

Open http://localhost:5000 in the browser.
"""

import argparse
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import openslide
import torch
from flask import Flask, abort, jsonify, render_template_string, send_file
from openslide.deepzoom import DeepZoomGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ABMIL.create_splits import COMPARISONS
from src.ABMIL.heatmap import PATCH_SIZE_UM, class_label, load_model, run_inference

app = Flask(__name__)

# populated in __main__
SLIDE_DIR: Path = None
PRED_DIR: Path = None
FEATURES_DIR: Path = None
MIL_DIR: Path = None
TILE_SIZE = 256
slides_cache: dict = {}
mil_models: dict = {}           # comparison -> loaded model
attention_cache: dict = {}      # (slide, comparison) -> response dict
slide_path_group: dict = {}     # slide stem -> ground-truth pathology group


def load_path_groups(xlsx_path: Path) -> dict:
    """Map slide stem -> path_group label straight from the spreadsheet."""
    import pandas as pd
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.columns = [
        "case_id", "age", "block_id", "stain", "description",
        "scanner", "mag", "filename", "path_group",
        *[f"extra_{i}" for i in range(len(df.columns) - 9)],
    ]
    df = df.dropna(subset=["filename", "path_group"])
    return {str(r["filename"]).replace(".svs", ""): str(r["path_group"])
            for _, r in df.iterrows()}


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


GROUP_ORDER = ["Control", "RHI", "Low CTE", "High CTE"]


@app.route("/")
def index():
    stems = sorted(p.stem for p in SLIDE_DIR.glob("*.svs"))
    groups: dict = {}
    for s in stems:
        groups.setdefault(slide_path_group.get(s) or "Unlabeled", []).append(s)
    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    ordered = sorted(
        groups.items(),
        key=lambda kv: (rank.get(kv[0], 99), kv[0] == "Unlabeled", kv[0]),
    )
    return render_template_string(INDEX_HTML, groups=ordered, total=len(stems))


@app.route("/<name>/")
def viewer(name):
    return render_template_string(
        VIEWER_HTML, name=name, label=slide_path_group.get(name, "")
    )


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


def available_comparisons():
    if MIL_DIR is None:
        return []
    return [c for c in COMPARISONS if (MIL_DIR / f"mil_{c}.pth").exists()]


def get_mil_model(comparison):
    if comparison not in mil_models:
        ckpt_path = MIL_DIR / f"mil_{comparison}.pth"
        if not ckpt_path.exists():
            return None
        print(f"[mil] loading {ckpt_path}")
        mil_models[comparison], _ = load_model(ckpt_path)
    return mil_models[comparison]


@app.route("/comparisons.json")
def comparisons_route():
    return jsonify([
        {"key": c,
         "name": COMPARISONS[c]["name"],
         "class_0": class_label(COMPARISONS[c]["class_0"]),
         "class_1": class_label(COMPARISONS[c]["class_1"])}
        for c in available_comparisons()
    ])


@app.route("/<name>/attention/<comparison>.json")
def attention(name, comparison):
    if comparison not in COMPARISONS:
        abort(404, "unknown comparison")
    key = (name, comparison)
    if key in attention_cache:
        return jsonify(attention_cache[key])

    if FEATURES_DIR is None:
        abort(404, "features-dir not configured")
    feat_path = FEATURES_DIR / f"{name}.pt"
    if not feat_path.exists():
        abort(404, f"features not found: {feat_path}")

    model = get_mil_model(comparison)
    if model is None:
        abort(404, f"checkpoint not found for {comparison}")

    data = torch.load(feat_path, map_location="cpu")
    features = data["features"]
    coords_raw = data["coords"]
    coords = coords_raw.numpy() if torch.is_tensor(coords_raw) else np.asarray(coords_raw)
    mpp = float(data["mpp"])

    probs, att = run_inference(model, features)
    ranks = np.argsort(np.argsort(att)) / max(len(att) - 1, 1)
    tile_px = int(PATCH_SIZE_UM / mpp)
    comp = COMPARISONS[comparison]
    names = [class_label(comp["class_0"]), class_label(comp["class_1"])]
    pred = int(np.argmax(probs))

    result = {
        "tile_px": tile_px,
        "prediction": {
            "pred": pred,
            "names": names,
            "probs": [float(probs[0]), float(probs[1])],
        },
        "tiles": [
            {"x": int(x), "y": int(y), "a": float(a)}
            for (x, y), a in zip(coords, ranks)
        ],
    }
    attention_cache[key] = result
    return jsonify(result)


INDEX_HTML = """
<!doctype html>
<title>VascuPath slides</title>
<style>
  body   { font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; }
  h2     { margin: 1.6rem 0 0.4rem; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  h2 .n  { color: #888; font-weight: normal; font-size: 0.8em; margin-left: 6px; }
  ul     { line-height: 1.7; list-style: none; padding: 0;
           columns: 3; column-gap: 1.5rem; }
  li     { break-inside: avoid; }
  a      { text-decoration: none; }
</style>
<h1>Slides ({{total}})</h1>
{% for group, stems in groups %}
  <h2>{{group}} <span class="n">({{stems|length}})</span></h2>
  <ul>
  {% for s in stems %}
    <li><a href="/{{s}}/">{{s}}</a></li>
  {% endfor %}
  </ul>
{% else %}
  <p><em>no slides found</em></p>
{% endfor %}
"""


VIEWER_HTML = """
<!doctype html>
<title>{{name}}</title>
<script src="https://cdn.jsdelivr.net/npm/openseadragon@4/build/openseadragon/openseadragon.min.js"></script>
<style>
  body { margin: 0; font-family: system-ui, sans-serif; }
  #osd { width: 100vw; height: 100vh; background: #111; }
  #hud {
    position: fixed; top: 10px; left: 10px; padding: 8px 12px;
    background: rgba(0,0,0,0.65); color: #fff; font-size: 12px; border-radius: 4px;
    max-width: 380px;
  }
  #hud .row { margin-top: 4px; }
  label { margin-right: 12px; cursor: pointer; white-space: nowrap; }
  .status { color: #aaa; font-style: italic; margin-left: 6px; }
  body.hide-vessels .overlay-vessel { display: none !important; }
  /* per-comparison attention layers — toggled by adding hide-att-<key> on body */
</style>
<div id="osd"></div>
<div id="hud">
  <div>
    <strong>{{name}}</strong>
    {% if label %}<span style="color:#fc9">— {{label}}</span>{% endif %}
    &nbsp;<a href="/" style="color:#9cf">← all slides</a>
  </div>
  <div class="row"><label><input type="checkbox" id="toggle-vessels" checked> vessels</label></div>
  <div id="attention-toggles"></div>
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

// Jet-style colormap, t in [0,1] -> [r,g,b] in [0,255].
function jet(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 3))));
  const g = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 2))));
  const b = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 1))));
  return [r, g, b];
}

// Track which attention layers we've already rendered to avoid duplicate fetches.
const attentionLoaded = new Set();

v.addHandler("open", async () => {
  const img = v.world.getItemAt(0);

  // --- Vessel overlays ---
  const gj = await (await fetch("/" + SLIDE + "/predictions.geojson")).json();
  for (const f of gj.features) {
    const ring = f.geometry.coordinates[0];
    const xs = ring.map(c => c[0]), ys = ring.map(c => c[1]);
    const x = Math.min(...xs), y = Math.min(...ys);
    const w = Math.max(...xs) - x, h = Math.max(...ys) - y;
    const [r,g,b] = f.properties.classification.color;
    const cls    = f.properties.classification.name;

    const div = document.createElement("div");
    div.className = "overlay-vessel";
    div.title = cls;
    div.style.background = `rgba(${r},${g},${b},0.35)`;
    div.style.border     = `1px solid rgba(${r},${g},${b},0.9)`;
    v.addOverlay({ element: div, location: img.imageToViewportRectangle(x, y, w, h) });
  }

  // --- Attention checkbox row, one per available comparison ---
  const comparisons = await (await fetch("/comparisons.json")).json();
  const togglesDiv = document.getElementById("attention-toggles");
  for (const c of comparisons) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `
      <label>
        <input type="checkbox" data-comp="${c.key}">
        attention: ${c.name}
      </label>
      <span class="status" id="status-${c.key}"></span>`;
    togglesDiv.appendChild(row);
    document.body.classList.add(`hide-att-${c.key}`);

    const styleId = `style-att-${c.key}`;
    if (!document.getElementById(styleId)) {
      const s = document.createElement("style");
      s.id = styleId;
      s.textContent = `body.hide-att-${c.key} .overlay-att-${c.key} { display: none !important; }`;
      document.head.appendChild(s);
    }
  }

  togglesDiv.addEventListener("change", async (e) => {
    const cb = e.target;
    if (cb.tagName !== "INPUT") return;
    const comp = cb.dataset.comp;
    document.body.classList.toggle(`hide-att-${comp}`, !cb.checked);

    if (cb.checked && !attentionLoaded.has(comp)) {
      attentionLoaded.add(comp);
      const status = document.getElementById(`status-${comp}`);
      status.textContent = "loading…";
      try {
        const r = await fetch(`/${SLIDE}/attention/${comp}.json`);
        if (!r.ok) {
          status.textContent = `(${r.status})`;
          attentionLoaded.delete(comp);
          return;
        }
        const data = await r.json();
        const px = data.tile_px;
        for (const t of data.tiles) {
          const [r,g,b] = jet(t.a);
          const div = document.createElement("div");
          div.className = `overlay-att-${comp}`;
          div.title = `${comp} attention=${t.a.toFixed(3)}`;
          div.style.background = `rgba(${r},${g},${b},0.45)`;
          v.addOverlay({ element: div, location: img.imageToViewportRectangle(t.x, t.y, px, px) });
        }
        const p = data.prediction;
        status.textContent = `→ ${p.names[p.pred]} (${p.probs[p.pred].toFixed(2)})`;
      } catch (err) {
        status.textContent = "(error)";
        attentionLoaded.delete(comp);
      }
    }
  });
});

document.getElementById("toggle-vessels").addEventListener("change", e => {
  document.body.classList.toggle("hide-vessels", !e.target.checked);
});
</script>
"""


def main():
    global SLIDE_DIR, PRED_DIR, FEATURES_DIR, MIL_DIR, slide_path_group
    ap = argparse.ArgumentParser(description="Browser-based WSI viewer")
    ap.add_argument("--slide-dir", required=True, type=Path,
                    help="Directory containing .svs files")
    ap.add_argument("--pred-dir",  required=True, type=Path,
                    help="Directory containing per-slide subfolders with predictions.geojson")
    ap.add_argument("--features-dir", type=Path, default=None,
                    help="Directory of <slide>.pt feature files (enables attention overlays)")
    ap.add_argument("--mil-dir", type=Path, default=None,
                    help="Directory of mil_<comparison>.pth checkpoints (enables attention overlays)")
    ap.add_argument("--labels-xlsx", type=Path, default=Path("data/case_labels.xlsx"),
                    help="Label spreadsheet; ground-truth path_group shown beside each slide")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    SLIDE_DIR    = args.slide_dir.resolve()
    PRED_DIR     = args.pred_dir.resolve()
    FEATURES_DIR = args.features_dir.resolve() if args.features_dir else None
    MIL_DIR      = args.mil_dir.resolve() if args.mil_dir else None

    if args.labels_xlsx and args.labels_xlsx.exists():
        slide_path_group = load_path_groups(args.labels_xlsx.resolve())
        print(f"Labels       : {args.labels_xlsx} ({len(slide_path_group)} entries)")
    else:
        print(f"Labels       : (not found at {args.labels_xlsx})")

    print(f"Slide dir    : {SLIDE_DIR}")
    print(f"Pred dir     : {PRED_DIR}")
    print(f"Features dir : {FEATURES_DIR}")
    print(f"MIL dir      : {MIL_DIR}")
    if MIL_DIR is not None:
        print(f"Available comparisons: {available_comparisons() or '(none found)'}")
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
