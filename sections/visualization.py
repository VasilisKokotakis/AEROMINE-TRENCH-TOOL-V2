import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd


def _decimate(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    f = 1 if not factor or factor < 1 else int(factor)
    return df.iloc[::f, :].reset_index(drop=True)


def create_3d_html(df: pd.DataFrame, decimate_factor: int = 1) -> None:
    df3d = _decimate(df, decimate_factor)
    fig3d = px.scatter_3d(
        df3d, x="x", y="y", z="z", color="section_id", opacity=0.7
    )
    fig3d.update_layout(title="3D trench preview", height=800)
    fig3d.write_html("sections_3d.html")
    print(" -> Saved sections_3d.html")


def create_2d_html(df: pd.DataFrame) -> None:
    # Ελαφρύ decimate για 2D
    df2d = _decimate(df, 50)

    section_ids = sorted(df2d["section_id"].unique().tolist())
    if not section_ids:
        return

    # Προετοιμασία δεδομένων ανά τομή για JS
    sections_js = []
    for sid in section_ids:
        sub = df2d[df2d["section_id"] == sid]
        xs = [float(v) for v in sub["dist_off"].to_numpy()]
        zs = [float(v) for v in sub["z"].to_numpy()]
        sections_js.append(
            {
                "id": int(sid),
                "zmin": float(sub["z"].min()),
                "zmax": float(sub["z"].max()),
                "x_min": float(sub["dist_off"].min()),
                "x_max": float(sub["dist_off"].max()),
                "xs": xs,
                "zs": zs,
            }
        )

    # Αρχική τομή
    first_sid = section_ids[0]
    first = [s for s in sections_js if s["id"] == first_sid][0]

    scatter = go.Scattergl(
        x=first["xs"],
        y=first["zs"],
        mode="markers",
        marker=dict(size=2, opacity=0.7),
        name=f"Section {first_sid}",
    )

    # Σχήματα:
    # 0: οριζόντια Zmin (xref:x, yref:y)
    # 1: οριζόντια Zmax (xref:x, yref:y)
    # 2: κατακόρυφη Xmin (xref:x, yref:paper)
    # 3: κατακόρυφη Xmax (xref:x, yref:paper)
    shapes = [
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=first["x_min"],
            x1=first["x_max"],
            y0=first["zmin"],
            y1=first["zmin"],
            line=dict(dash="dot", width=2),
        ),
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=first["x_min"],
            x1=first["x_max"],
            y0=first["zmax"],
            y1=first["zmax"],
            line=dict(width=2),
        ),
        dict(  # Xmin vertical
            type="line",
            xref="x",
            yref="paper",
            x0=first["x_min"],
            x1=first["x_min"],
            y0=0,
            y1=1,
            line=dict(dash="dot", width=2),
        ),
        dict(  # Xmax vertical
            type="line",
            xref="x",
            yref="paper",
            x0=first["x_max"],
            x1=first["x_max"],
            y0=0,
            y1=1,
            line=dict(width=2),
        ),
    ]

    # Σχολιασμοί: 0 Zmin, 1 Zmax, 2 Xmin, 3 Xmax
    annotations = [
        dict(
            x=first["x_max"],
            y=first["zmin"],
            xref="x",
            yref="y",
            text=f"Zmin={first['zmin']:.3f} m",
            showarrow=False,
            xanchor="left",
        ),
        dict(
            x=first["x_max"],
            y=first["zmax"],
            xref="x",
            yref="y",
            text=f"Zmax={first['zmax']:.3f} m",
            showarrow=False,
            xanchor="left",
        ),
        dict(
            x=first["x_min"],
            y=0.02,
            xref="x",
            yref="paper",
            text=f"Xmin={first['x_min']:.3f} m",
            showarrow=False,
            xanchor="center",
        ),
        dict(
            x=first["x_max"],
            y=0.98,
            xref="x",
            yref="paper",
            text=f"Xmax={first['x_max']:.3f} m",
            showarrow=False,
            xanchor="center",
        ),
    ]

    fig2d = go.Figure(data=[scatter])
    fig2d.update_layout(
        title=f"Section {first_sid} (2D)",
        xaxis_title="Offset (m)",
        yaxis_title="Elevation (Z, m)",
        height=700,
        shapes=shapes,
        annotations=annotations,
    )

    fig_html = pio.to_html(
        fig2d, full_html=False, include_plotlyjs="cdn", div_id="sections2d"
    )
    sections_json = json.dumps(sections_js)

    # Τελική HTML με ΑΜΕΣΗ αλλαγή τομής στο dropdown (input/change)
    html_page = (
        "<!doctype html>\n<html>\n<head>\n<meta charset='utf-8'>\n<title>sections_2d</title>\n"
        "<style>\n"
        "body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial; margin:0}\n"
        "#wrap{display:grid; grid-template-columns:1fr 320px; gap:12px; padding:10px}\n"
        "#controls{border-left:1px solid #eee; padding-left:12px}\n"
        "label{display:block; margin:6px 0 2px; font-weight:600}\n"
        "input[type=number]{width:100%; padding:6px}\n"
        "button{margin-top:8px; padding:8px 12px; cursor:pointer}\n"
        ".small{color:#666; font-size:12px; margin-top:6px}\n"
        ".metric{font-weight:700; margin-top:4px}\n"
        "</style>\n</head>\n<body>\n<div id='wrap'>\n"
        f"{fig_html}\n"
        "<div id='controls'>\n"
        "<h3>Adjust Lines</h3>\n"
        "<div class='small'>Per-section · values persist while this page is open</div>\n"
        "<label for='sectionSelect'>Section</label>\n"
        "<select id='sectionSelect'></select>\n"
        "<label for='zminInput'>Zmin (m)</label>\n"
        "<input id='zminInput' type='number' step='0.01' />\n"
        "<label for='zmaxInput'>Zmax (m)</label>\n"
        "<input id='zmaxInput' type='number' step='0.01' />\n"
        "<label for='xminInput'>Xmin (m)</label>\n"
        "<input id='xminInput' type='number' step='0.01' />\n"
        "<label for='xmaxInput'>Xmax (m)</label>\n"
        "<input id='xmaxInput' type='number' step='0.01' />\n"
        "<button id='applyBtn'>Apply</button>\n"
        "<button id='resetBtn'>Reset</button>\n"
        "<div class='small'>Heights & Widths</div>\n"
        "<div id='metricsInfo' class='metric'>—</div>\n"
        "</div>\n</div>\n"
        "<script>\n"
        "const sections = " + sections_json + ";\n"
        "const gd = document.getElementById('sections2d');\n"
        "const select = document.getElementById('sectionSelect');\n"
        "const zminInput = document.getElementById('zminInput');\n"
        "const zmaxInput = document.getElementById('zmaxInput');\n"
        "const xminInput = document.getElementById('xminInput');\n"
        "const xmaxInput = document.getElementById('xmaxInput');\n"
        "const applyBtn = document.getElementById('applyBtn');\n"
        "const resetBtn = document.getElementById('resetBtn');\n"
        "const metricsInfo = document.getElementById('metricsInfo');\n"
        "let store = {}; // per-section overrides, π.χ. store[idx] = { zmin, zmax, xmin, xmax }\n"
        "\n"
        "function setSelectOptions(){\n"
        "  sections.forEach((s,i)=>{\n"
        "    const opt = document.createElement('option');\n"
        "    opt.value = i; opt.text = 'Section ' + s.id;\n"
        "    select.appendChild(opt);\n"
        "  });\n"
        "}\n"
        "\n"
        "function loadSection(idx){\n"
        "  const s = sections[idx];\n"
        "  // Ενημέρωση σημείων ΑΜΕΣΑ\n"
        "  Plotly.restyle(gd, { 'x': [s.xs], 'y': [s.zs] }, [0]);\n"
        "  // Ανεξάρτητες τιμές Z και X (με override όπου υπάρχει)\n"
        "  const zmin = (store[idx] && Number.isFinite(store[idx].zmin)) ? store[idx].zmin : s.zmin;\n"
        "  const zmax = (store[idx] && Number.isFinite(store[idx].zmax)) ? store[idx].zmax : s.zmax;\n"
        "  const xmin = (store[idx] && Number.isFinite(store[idx].xmin)) ? store[idx].xmin : s.x_min;\n"
        "  const xmax = (store[idx] && Number.isFinite(store[idx].xmax)) ? store[idx].xmax : s.x_max;\n"
        "  // Relayout μόνο των αντίστοιχων στοιχείων (χωρίς διασταυρούμενη επίδραση)\n"
        "  const rel = {\n"
        "    'shapes[0].x0': s.x_min, 'shapes[0].x1': s.x_max, 'shapes[0].y0': zmin, 'shapes[0].y1': zmin,\n"
        "    'shapes[1].x0': s.x_min, 'shapes[1].x1': s.x_max, 'shapes[1].y0': zmax, 'shapes[1].y1': zmax,\n"
        "    'shapes[2].x0': xmin,    'shapes[2].x1': xmin,    'shapes[2].y0': 0,    'shapes[2].y1': 1,\n"
        "    'shapes[3].x0': xmax,    'shapes[3].x1': xmax,    'shapes[3].y0': 0,    'shapes[3].y1': 1,\n"
        "    'annotations[0].x': s.x_max, 'annotations[0].y': zmin, 'annotations[0].text': 'Zmin=' + zmin.toFixed(3) + ' m',\n"
        "    'annotations[1].x': s.x_max, 'annotations[1].y': zmax, 'annotations[1].text': 'Zmax=' + zmax.toFixed(3) + ' m',\n"
        "    'annotations[2].x': xmin,    'annotations[2].y': 0.02, 'annotations[2].text': 'Xmin=' + xmin.toFixed(3) + ' m',\n"
        "    'annotations[3].x': xmax,    'annotations[3].y': 0.98, 'annotations[3].text': 'Xmax=' + xmax.toFixed(3) + ' m'\n"
        "  };\n"
        "  Plotly.relayout(gd, rel);\n"
        "  // Inputs\n"
        "  zminInput.value = zmin.toFixed(3);\n"
        "  zmaxInput.value = zmax.toFixed(3);\n"
        "  xminInput.value = xmin.toFixed(3);\n"
        "  xmaxInput.value = xmax.toFixed(3);\n"
        "  // Μετρικά\n"
        "  metricsInfo.textContent = 'Δz = ' + (zmax - zmin).toFixed(3) + ' m,  Δx = ' + (xmax - xmin).toFixed(3) + ' m';\n"
        "}\n"
        "\n"
        "// Αλλαγή τομής άμεσα (δουλεύει και με τα ↓/↑ του πληκτρολογίου)\n"
        "select.addEventListener('input', () => { loadSection(parseInt(select.value)); });\n"
        "select.addEventListener('change', () => { loadSection(parseInt(select.value)); });\n"
        "\n"
        "// Apply: αποθηκεύει overrides ΜΟΝΟ για την τρέχουσα τομή και ξαναφορτώνει\n"
        "applyBtn.addEventListener('click', () => {\n"
        "  const idx = parseInt(select.value);\n"
        "  store[idx] = {\n"
        "    zmin: parseFloat(zminInput.value),\n"
        "    zmax: parseFloat(zmaxInput.value),\n"
        "    xmin: parseFloat(xminInput.value),\n"
        "    xmax: parseFloat(xmaxInput.value)\n"
        "  };\n"
        "  loadSection(idx);\n"
        "});\n"
        "\n"
        "// Reset: διαγράφει overrides για την τρέχουσα τομή και επιστρέφει στα default\n"
        "resetBtn.addEventListener('click', () => {\n"
        "  const idx = parseInt(select.value);\n"
        "  delete store[idx];\n"
        "  loadSection(idx);\n"
        "});\n"
        "\n"
        "// Init\n"
        "setSelectOptions();\n"
        "select.value = 0;\n"
        "loadSection(0);\n"
        "</script>\n</body>\n</html>\n"
    )

    with open("sections_2d.html", "w", encoding="utf-8") as f:
        f.write(html_page)
    print(" -> Saved sections_2d.html")
