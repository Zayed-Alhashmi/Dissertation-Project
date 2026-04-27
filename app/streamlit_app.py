import sys, os, tempfile, io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="CAC Score Analyser", page_icon="❤",
                   layout="wide", initial_sidebar_state="expanded")

# Inject raw CSS to forcefully override Streamlit's native component styling.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800&display=swap');

*, html, body { font-family: 'Inter', sans-serif !important; }

/* App background */
.stApp { background: linear-gradient(145deg, #eef2f7 0%, #e8edf5 100%) !important; }
.block-container { padding: 0 2.5rem 3rem 2.5rem !important; max-width: 1080px !important; }
.element-container { margin-bottom: 0 !important; }
section.main > div { padding-top: 0 !important; }

/* -- Sidebar -- */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] * { color: #334155 !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem !important; }
[data-testid="stSidebar"] > div { padding-top: 0.5rem !important; }
[data-testid="stSidebarContent"] { padding-top: 0.5rem !important; }
section[data-testid="stSidebar"] > div:first-child > div { padding-top: 0 !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
/* Reduce Streamlit default sidebar top padding */
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem !important; margin-top: 0 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2 { color: #0f172a !important; font-weight: 700 !important; }
[data-testid="stSidebar"] hr { border-color: #f1f5f9 !important; }
[data-testid="stSidebar"] a { color: #2563eb !important; text-decoration: none !important; }
[data-testid="stSidebar"] label { color: #475569 !important; font-size: 13px !important; }
section[data-testid="collapsedControl"] { background: #ffffff !important; border-right: 1px solid #e2e8f0 !important; }
section[data-testid="collapsedControl"] svg { fill: #64748b !important; }

/* -- Risk badges -- */
.rbadge { display:inline-flex; align-items:center; padding:6px 20px; border-radius:100px;
          font-weight:600; font-size:13px; letter-spacing:0.02em; gap:6px; }
.rbadge-green  { background:#dcfce7; color:#15803d; }
.rbadge-yellow { background:#fef9c3; color:#92400e; }
.rbadge-orange { background:#ffedd5; color:#9a3412; }
.rbadge-red    { background:#fee2e2; color:#991b1b; }

/* -- Step number pill -- */
.step-pill {
    display: inline-flex; align-items: center; gap: 10px;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white; border-radius: 100px;
    padding: 6px 16px 6px 8px;
    font-size: 13px; font-weight: 700;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3);
    margin-bottom: 12px;
}
.step-pill-num {
    background: rgba(255,255,255,0.25); border-radius: 50%;
    width: 22px; height: 22px; display: inline-flex;
    align-items: center; justify-content: center; font-size: 12px;
}

/* -- Buttons -- */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 15px !important; padding: 12px 28px !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
    transition: all 0.2s !important; letter-spacing: 0.01em !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.45) !important;
}

/* -- File uploader -- */
[data-testid="stFileUploader"] section {
    background: #f8faff !important;
    border: 2px dashed #93c5fd !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}

/* -- Progress bar -- */
[data-testid="stProgressBar"] > div { background: #eff6ff !important; border-radius: 100px !important; }
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg,#2563eb,#60a5fa) !important; border-radius:100px !important; }

/* -- Hide chrome -- */
#MainMenu,footer,[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="stStatusWidget"] { display:none !important; }
[data-testid="StyledLinkIconContainer"] svg { display:none !important; }
div.element-container:hover { outline:none !important; background:transparent !important; }

/* Uploaded file list - force all text black */
[data-testid="stFileUploader"] * { color: #0f172a !important; }
[data-testid="stFileUploader"] li { color: #0f172a !important; }
[data-testid="stFileUploader"] p  { color: #0f172a !important; }
[data-testid="stFileUploader"] small { color: #374151 !important; }
[data-testid="stFileUploader"] span { color: #0f172a !important; }
[data-testid="stFileUploader"] div { color: #0f172a !important; }
/* Restore Browse files button text to white */
[data-testid="stFileUploader"] button { color: white !important; }
[data-testid="stFileUploader"] button * { color: white !important; }
/* Better Selectbox and UI overrides */
[data-testid="stSidebar"] div[data-baseweb="select"] { margin-top: 10px !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div { background: #f8fafc !important; border: 1px solid #cbd5e1 !important; color: #0f172a !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] span { color: #0f172a !important; }
[data-testid="stSidebar"] ul { background-color: #ffffff !important; }
[data-testid="stSidebar"] li { color: #0f172a !important; }

/* Tabs text overrides */
[data-testid="stMarkdownContainer"] { color: #334155; }
[data-testid="stTab"] { color: #0f172a !important; font-weight: 600 !important; }
[data-testid="stTab"][aria-selected="true"] { border-bottom-color: #2563eb !important; }
/* Hide Streamlit's ugly hover anchors on headers */
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a, .header-anchor, a[href^="#"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# MESA reference data (Hoff 2001 / McClelland MESA 2006)
MESA = {
    ("M","White",45):{"z":65,"bp":{25:0,50:0,75:27,90:126,95:231,99:1000}},
    ("M","White",50):{"z":50,"bp":{25:0,50:7,75:97,90:279,95:431,99:1600}},
    ("M","White",55):{"z":37,"bp":{25:0,50:44,75:168,90:476,95:731,99:2100}},
    ("M","White",60):{"z":28,"bp":{25:0,50:80,75:302,90:625,95:900,99:2400}},
    ("M","White",65):{"z":21,"bp":{25:0,50:82,75:371,90:792,95:1100,99:2800}},
    ("M","White",70):{"z":15,"bp":{25:12,50:175,75:547,90:1082,95:1500,99:3500}},
    ("M","White",75):{"z":11,"bp":{25:24,50:240,75:639,90:1197,95:1700,99:3800}},
    ("F","White",45):{"z":87,"bp":{25:0,50:0,75:0,90:13,95:46,99:200}},
    ("F","White",50):{"z":75,"bp":{25:0,50:0,75:0,90:70,95:149,99:580}},
    ("F","White",55):{"z":60,"bp":{25:0,50:0,75:30,90:149,95:267,99:900}},
    ("F","White",60):{"z":48,"bp":{25:0,50:0,75:73,90:305,95:524,99:1600}},
    ("F","White",65):{"z":37,"bp":{25:0,50:3,75:124,90:406,95:639,99:1900}},
    ("F","White",70):{"z":27,"bp":{25:0,50:47,75:259,90:634,95:946,99:2500}},
    ("F","White",75):{"z":20,"bp":{25:0,50:67,75:298,90:678,95:1050,99:2750}},
    ("M","Black",45):{"z":72,"bp":{25:0,50:0,75:14,90:108,95:205,99:840}},
    ("M","Black",50):{"z":57,"bp":{25:0,50:0,75:58,90:233,95:382,99:1150}},
    ("M","Black",55):{"z":45,"bp":{25:0,50:14,75:132,90:390,95:610,99:1700}},
    ("M","Black",60):{"z":35,"bp":{25:0,50:38,75:228,90:532,95:790,99:2000}},
    ("M","Black",65):{"z":26,"bp":{25:0,50:66,75:330,90:710,95:1020,99:2500}},
    ("M","Black",70):{"z":19,"bp":{25:0,50:140,75:488,90:960,95:1380,99:3200}},
    ("M","Black",75):{"z":14,"bp":{25:10,50:228,75:580,90:1130,95:1600,99:3600}},
    ("F","Black",45):{"z":90,"bp":{25:0,50:0,75:0,90:6,95:25,99:146}},
    ("F","Black",50):{"z":80,"bp":{25:0,50:0,75:0,90:40,95:100,99:440}},
    ("F","Black",55):{"z":67,"bp":{25:0,50:0,75:13,90:104,95:201,99:750}},
    ("F","Black",60):{"z":54,"bp":{25:0,50:0,75:56,90:233,95:410,99:1300}},
    ("F","Black",65):{"z":43,"bp":{25:0,50:0,75:130,90:380,95:610,99:1700}},
    ("F","Black",70):{"z":33,"bp":{25:0,50:26,75:248,90:592,95:900,99:2300}},
    ("F","Black",75):{"z":24,"bp":{25:0,50:64,75:326,90:766,95:1150,99:2800}},
    ("M","Hispanic",45):{"z":74,"bp":{25:0,50:0,75:10,90:88,95:175,99:690}},
    ("M","Hispanic",50):{"z":58,"bp":{25:0,50:0,75:44,90:188,95:312,99:950}},
    ("M","Hispanic",55):{"z":46,"bp":{25:0,50:9,75:108,90:314,95:498,99:1380}},
    ("M","Hispanic",60):{"z":36,"bp":{25:0,50:30,75:196,90:446,95:678,99:1750}},
    ("M","Hispanic",65):{"z":27,"bp":{25:0,50:54,75:292,90:614,95:890,99:2100}},
    ("M","Hispanic",70):{"z":20,"bp":{25:0,50:116,75:432,90:870,95:1250,99:2900}},
    ("M","Hispanic",75):{"z":15,"bp":{25:8,50:212,75:536,90:1060,95:1520,99:3400}},
    ("F","Hispanic",45):{"z":91,"bp":{25:0,50:0,75:0,90:5,95:19,99:120}},
    ("F","Hispanic",50):{"z":82,"bp":{25:0,50:0,75:0,90:31,95:84,99:360}},
    ("F","Hispanic",55):{"z":70,"bp":{25:0,50:0,75:8,90:90,95:176,99:630}},
    ("F","Hispanic",60):{"z":57,"bp":{25:0,50:0,75:44,90:196,95:348,99:1100}},
    ("F","Hispanic",65):{"z":45,"bp":{25:0,50:0,75:110,90:326,95:530,99:1450}},
    ("F","Hispanic",70):{"z":35,"bp":{25:0,50:20,75:224,90:546,95:820,99:2100}},
    ("F","Hispanic",75):{"z":26,"bp":{25:0,50:55,75:296,90:700,95:1060,99:2600}},
    ("M","Chinese",45):{"z":76,"bp":{25:0,50:0,75:8,90:72,95:152,99:580}},
    ("M","Chinese",50):{"z":63,"bp":{25:0,50:2,75:34,90:150,95:264,99:810}},
    ("M","Chinese",55):{"z":50,"bp":{25:0,50:6,75:86,90:248,95:404,99:1100}},
    ("M","Chinese",60):{"z":40,"bp":{25:0,50:24,75:162,90:370,95:570,99:1480}},
    ("M","Chinese",65):{"z":31,"bp":{25:0,50:44,75:250,90:534,95:780,99:1900}},
    ("M","Chinese",70):{"z":23,"bp":{25:0,50:100,75:378,90:778,95:1120,99:2600}},
    ("M","Chinese",75):{"z":17,"bp":{25:6,50:190,75:476,90:952,95:1380,99:3100}},
    ("F","Chinese",45):{"z":93,"bp":{25:0,50:0,75:0,90:4,95:14,99:90}},
    ("F","Chinese",50):{"z":85,"bp":{25:0,50:0,75:0,90:22,95:64,99:280}},
    ("F","Chinese",55):{"z":74,"bp":{25:0,50:0,75:5,90:70,95:148,99:510}},
    ("F","Chinese",60):{"z":62,"bp":{25:0,50:0,75:36,90:164,95:296,99:940}},
    ("F","Chinese",65):{"z":50,"bp":{25:0,50:0,75:92,90:280,95:460,99:1240}},
    ("F","Chinese",70):{"z":40,"bp":{25:0,50:16,75:196,90:480,95:726,99:1850}},
    ("F","Chinese",75):{"z":30,"bp":{25:0,50:46,75:268,90:640,95:970,99:2400}},
}


def get_avg_ref(sex_label, age):
    sx = "M" if sex_label == "Male" else "F"
    age_lo = max(45, min(75, (age // 5) * 5))
    groups = [MESA.get((sx, r, age_lo)) for r in ["White","Black","Hispanic","Chinese"]]
    groups = [g for g in groups if g]
    if not groups:
        return None
    avg_z  = np.mean([g["z"] for g in groups])
    avg_bp = {p: int(np.mean([g["bp"][p] for g in groups if p in g["bp"]]))
              for p in [25, 50, 75, 90, 95, 99]}
    return {"z": avg_z, "bp": avg_bp}


def score_to_percentile(score, ref):
    z, bp = ref["z"], ref["bp"]
    if score <= 0:
        return z / 2
    xs = [0.0]
    ys = [float(z)]
    for pct in sorted(bp.keys()):
        s = float(bp[pct])
        if pct >= z and s > xs[-1]:
            xs.append(s); ys.append(float(pct))
    if score >= xs[-1]:
        return min(99.9, ys[-1] + 1.0)
    return float(np.interp(score, xs, ys))


def make_percentile_chart(score, pct, sex_label, age, ref):
    bp = ref["bp"]
    fig = plt.figure(figsize=(10, 3.0), facecolor="white")
    ax  = fig.add_axes([0.05, 0.28, 0.90, 0.42])

    # Gradient bar
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", extent=[0, 100, 0, 1],
              cmap="RdYlGn_r", alpha=0.88, zorder=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.9, 2.6)

    # Zone dividers
    for x in [25, 50, 75, 90]:
        ax.axvline(x, color="white", lw=1.8, alpha=0.9, zorder=2)

    # Patient marker
    p = min(max(float(pct), 0.8), 99.2)
    ax.plot([p, p], [-0.05, 1.05], color="#0f172a", lw=2.5, zorder=4,
            solid_capstyle="round", clip_on=False)
    ax.plot(p, 1.08, "v", color="#0f172a", ms=11, zorder=5, clip_on=False)
    label_x = min(max(p, 8), 92)
    ax.text(label_x, 1.55, f"{pct:.0f}th percentile",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#0f172a")

    # Zone labels
    for x, lbl in [(12.5,"Low"),(37.5,"Below avg"),(62.5,"Above avg"),(82.5,"High"),(95,"Very high")]:
        ax.text(x, -0.28, lbl, ha="center", va="top", fontsize=8, color="#64748b")

    # Reference scores at percentile ticks
    clrs = {25:"#16a34a", 50:"#ca8a04", 75:"#ea580c", 90:"#dc2626"}
    for rp in [25, 50, 75, 90]:
        rs = bp.get(rp, 0)
        ax.text(rp, 2.3, f"≤{rs}", ha="center", va="top",
                fontsize=9, fontweight="600", color=clrs[rp])
        ax.text(rp, 1.95, f"p{rp}", ha="center", va="top",
                fontsize=7.5, color="#94a3b8", fontweight="500")

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.text(0.5, 0.97,
             f"Population Reference  ·  {sex_label}, Age {age}  ·  (averaged across major demographic groups)",
             ha="center", fontsize=9.5, color="#64748b", style="italic")
    fig.text(0.5, 0.07,
             "Source: Hoff JA et al. Am J Cardiol 2001; McClelland RL et al. Circulation 2006 (MESA)",
             ha="center", fontsize=7.5, color="#94a3b8")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# Risk helpers
RISK_LEVELS = [
    (  0,  0, "No Detectable Calcium",  "rbadge-green",  "#15803d", "#f0fdf4", "#bbf7d0",
       "Very Low Risk",
       "No calcium detected in the coronary arteries. Associated with a very low 10-year risk of major cardiovascular events. Standard preventive care is recommended."),
    (  1, 99, "Mild Calcification",     "rbadge-yellow", "#92400e", "#fefce8", "#fde68a",
       "Low–Moderate Risk",
       "Mild coronary calcification detected. Indicates early atherosclerosis. Lifestyle modifications and risk factor management are advised."),
    (100,399, "Moderate Calcification", "rbadge-orange", "#9a3412", "#fff7ed", "#fed7aa",
       "Intermediate Risk",
       "Moderate calcification detected. Associated with intermediate 10-year cardiovascular risk. Medical evaluation and preventive therapy are strongly recommended."),
    (400,None,"Severe Calcification",   "rbadge-red",    "#991b1b", "#fef2f2", "#fecaca",
       "High Risk",
       "Severe calcification detected. Associated with significantly elevated cardiovascular risk. Urgent medical evaluation and aggressive risk factor management are indicated."),
]

def get_risk(score):
    for lo, hi, label, badge, color, bg, border, short, text in RISK_LEVELS:
        if hi is None or score <= hi:
            return label, badge, color, bg, border, short, text
    return RISK_LEVELS[-1][2:]


def hu_to_uint8(hu):
    wc, ww = 50, 350
    vmin, vmax = wc - ww//2, wc + ww//2
    return ((np.clip(hu, vmin, vmax) - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def save_dcm_files(files):
    tmp = tempfile.mkdtemp()
    for f in files:
        with open(os.path.join(tmp, f.name), "wb") as out:
            out.write(f.getbuffer())
    return tmp


def run_pipeline(series, mode, arch):
    from classical.score_patient import process_slice
    from classical.utils import is_heart_level_slice, detect_aorta_circle
    n = len(series)
    start, end = int(0.20 * n), int(0.90 * n)
    aorta_cache = {}
    for i in range(start, end):
        _, hu, sp = series[i]
        if is_heart_level_slice(hu):
            r = detect_aorta_circle(hu, sp)
            if r is not None:
                aorta_cache[i] = r
    bar   = st.progress(0, text="Analysing slices…")
    total = 0.0
    slices = []
    for idx, i in enumerate(range(start, end)):
        bar.progress((idx + 1) / (end - start), text=f"Slice {idx+1} / {end-start}")
        path, hu, sp = series[i]
        mask, score = process_slice(hu, sp, mode=mode,
                                    slice_idx=i, aorta_cache=aorta_cache, arch=arch)
        if mask is None or score is None:
            continue
        total += score
        if score > 0:
            slices.append((os.path.basename(path), hu, score))
    bar.empty()
    slices.sort(key=lambda t: t[2], reverse=True)
    return total, slices


# Sidebar
with st.sidebar:
    st.markdown("""
<div style="padding:8px 0 4px 0;">
<div style="font-size:18px; font-weight:800; color:#0f172a;">CAC Analyser</div>
<div style="font-size:11px; color:#94a3b8; margin-top:2px;">Clinical Decision Support Tool</div>
</div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div style='font-size:11px;font-weight:700;color:#94a3b8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;'>Pipeline</div>", unsafe_allow_html=True)
    mode_label = st.radio("Scoring mode",
        ["Classical (HU Threshold)", "Hybrid (Classical + Deep Learning)"],
        index=1, label_visibility="collapsed")
    mode = "hybrid" if "Hybrid" in mode_label else "classical"
    
    arch = "resnet18"
    arch_label = "ResNet-18"
    if mode == "hybrid":
        st.markdown("<div style='font-size:11px;font-weight:700;color:#94a3b8;letter-spacing:0.1em;text-transform:uppercase;padding-top:20px;margin-bottom:8px;line-height:1.5;'>MODEL ARCHITECTURE</div><div style='height:8px;'></div>", unsafe_allow_html=True)
        arch_label = st.selectbox("Model", 
            ["ResNet-18", "EfficientNet-B0", "Custom CNN"], 
            index=0, label_visibility="collapsed")
        arch_map = {"ResNet-18": "resnet18", "EfficientNet-B0": "efficientnet_b0", "Custom CNN": "custom"}
        arch = arch_map[arch_label]

    st.markdown("---")
    st.markdown("<div style='font-size:11px;font-weight:700;color:#94a3b8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;'>Patient Demographics</div>", unsafe_allow_html=True)
    st.caption("Used for MESA percentile reference")
    pt_age = st.slider("Age", 40, 85, 60)
    pt_sex = st.radio("Sex", ["Male", "Female"])




# Renders the main title cardiovascular icon block.
st.markdown("""
<div style="background:white; border-radius:0 0 20px 20px;
            padding:28px 36px 22px; margin-bottom:28px;
            box-shadow:0 4px 24px rgba(37,99,235,0.08);
            border-bottom:3px solid #2563eb;">
<div style="display:flex; align-items:center; gap:18px;">
<div style="width:52px; height:52px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
      <img src="data:image/png;base64,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" width="48" height="48" style="display:block;" />
</div>
<div>
<div style="font-size:1.85rem; font-weight:800; color:#0f172a; line-height:1.1;
                  letter-spacing:-0.02em;">
        CAC Score Analyser
</div>
<div style="font-size:0.92rem; color:#64748b; margin-top:3px; font-weight:400;">
        Automated Coronary Artery Calcium Scoring from Chest CT
</div>
</div>
<div style="margin-left:auto; text-align:right;">
<div style="font-size:11px; color:#cbd5e1; font-weight:500;">PIPELINE</div>
<div style="font-size:13px; font-weight:600; color:#2563eb;">
        """ + ((f"Hybrid · {arch_label}") if mode == "hybrid" else "Classical") + """
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)


# Helper: card wrapper


def section_label(text):
    st.markdown(f"""<div style="font-size:11px; font-weight:700; color:#94a3b8;
    letter-spacing:0.1em; text-transform:uppercase; margin-bottom:14px;">{text}</div>""",
    unsafe_allow_html=True)



# App Content Tabs
tab_about, tab_guide, tab_scanner = st.tabs(["Home & About", "How to Use", "Run Scanner"])

with tab_about:
    st.markdown("""
<div style="background:white; border-radius:16px; padding:28px 32px; box-shadow:0 4px 15px rgba(0,0,0,0.03); border-top:4px solid #2563eb; margin-bottom:24px;">
<h3 style='margin-top:0; color:#0f172a; font-weight:800; font-size:22px;'>About the CAC Analyser</h3>
<div style='color:#475569; line-height:1.7; font-size:15px;'>
  The <strong style='color:#0f172a;'>Agatston score</strong> quantifies calcified coronary plaque from non-contrast chest CT scans. It is a powerful predictor of future cardiovascular events.<br><br>
  This clinical decision support tool automates the process of extracting, filtering, and scoring these calcifications natively from DICOM arrays, providing a comprehensive risk analysis for patients.
</div>
</div>

<div style="background:white; border-radius:16px; padding:28px 32px; box-shadow:0 4px 15px rgba(0,0,0,0.03); margin-bottom:24px;">
<h4 style='margin-top:0; color:#0f172a; font-weight:800; font-size:19px;'>Scoring Pipelines & Available Models</h4>
<div style='color:#475569; line-height:1.6; font-size:14px; margin-bottom:18px;'>
  This application offers two major scoring pipelines that you can select from the sidebar:
</div>
  
<div style='background:#f8fafc; border-left:4px solid #64748b; padding:16px 20px; border-radius:0 8px 8px 0; margin-bottom:16px;'>
<h5 style='margin-top:0; color:#334155; font-size:16px; font-weight:700;'>1. Classical Pipeline</h5>
<div style='color:#475569; font-size:14px; line-height:1.6;'>
    This mode implements the standard Agatston HU Thresholding (130 HU) and connected component aggregation. It strictly relies on anatomical heuristics (e.g. geometric bounding boxes, aortic exclusion circle) to reject false positive structures like bone and the aorta.
</div>
</div>

<div style='background:#eff6ff; border-left:4px solid #2563eb; padding:16px 20px; border-radius:0 8px 8px 0;'>
<h5 style='margin-top:0; color:#1e40af; font-size:16px; font-weight:700;'>2. Hybrid Pipeline (Classical + Deep Learning)</h5>
<div style='color:#475569; font-size:14px; line-height:1.6; margin-bottom:16px;'>
    The Classical pipeline is enhanced with a Deep Learning Convolutional Neural Network (CNN) filter. Candidates are extracted, converted into graphical pixel-patches, and classified using AI to robustly reject imaging noise, bone artifacts, and aortic calcifications.
</div>
    
<strong style='color:#1e40af; font-size:13px; text-transform:uppercase; letter-spacing:0.5px;'>Available Models</strong>
<ul style='margin-top:8px; color:#475569; font-size:13px; line-height:1.6;'>
<li style='margin-bottom:6px;'><strong style='color:#0f172a;'>ResNet-18 (Recommended):</strong> The most robust architecture. Uses residual skip-connections to learn deep spatial features without vanishing gradients. Highly accurate at discarding noise and spine false positives.</li>
<li style='margin-bottom:6px;'><strong style='color:#0f172a;'>EfficientNet-B0:</strong> An advanced, highly optimized architecture using compound scaling. Extremely fast inference times while maintaining high parameter efficiency. Great for lower-end hardware deployment.</li>
<li><strong style='color:#0f172a;'>Custom CNN:</strong> A bespoke, lightweight 3-layer convolutional network built from scratch specifically for this project. Demonstrates the fundamental theory of patch-classification but lacks the depth of ResNet.</li>
</ul>
</div>
</div>

<div style="background:white; border-radius:16px; padding:28px 32px; box-shadow:0 4px 15px rgba(0,0,0,0.03); margin-bottom:24px;">
<h4 style='margin-top:0; color:#0f172a; font-weight:800; font-size:19px;'>MESA Contextualisation</h4>
<div style='color:#475569; line-height:1.7; font-size:14px;'>
  Absolute CAC burden means very little clinically without demographic context. A score of 200 in a 45-year old female implies severe cardiovascular risk, while the same score in an 85-year old male is below average.<br><br>
  This tool linearly interpolates patient scores against the <b>Multi-Ethnic Study of Atherosclerosis (MESA)</b> and Hoff 2001 population averages by Age and Sex, surfacing the exact percentile risk.
</div>
</div>
""", unsafe_allow_html=True)

with tab_guide:
    st.markdown("""
<div style="background:white; border-radius:16px; padding:28px 32px; box-shadow:0 4px 15px rgba(0,0,0,0.03); border-top:4px solid #10b981; margin-bottom:24px;">
<h3 style='margin-top:0; color:#0f172a; font-weight:800; font-size:22px;'>How to Use the Scanner</h3>
<div style='color:#475569; font-size:15px;'>
<ol style='line-height:2.0; padding-left:20px;'>
<li><strong style='color:#0f172a;'>Configure Patient Demographics:</strong><br>Use the sidebar on the left to set the patient's exact <b>Age</b> and <b>Sex</b>. This is tightly linked to the MESA reference percentile.</li>
<li><strong style='color:#0f172a;'>Select the Pipeline:</strong><br>Choose either <b>Classical</b> or <b>Hybrid</b>. If Hybrid is selected, pick the desired CNN Architecture from the dropdown menu below it.</li>
<li><strong style='color:#0f172a;'>Upload DICOM Files:</strong><br>Switch to the <b>Run Scanner</b> tab. Click <code>Browse files</code> and select <b>all <code>*.dcm</code> files</b> (Cmd+A / Ctrl+A) belonging to one patient series at once. The scanner expects axial non-contrast CT sequences.</li>
<li><strong style='color:#0f172a;'>Execute Analysis:</strong><br>Click the blue <b>Run Pipeline</b> button. The application will process the slices, compute Agatston scores dynamically, and produce an anatomical slice-level breakdown table and global risk score.</li>
</ol>
</div>
</div>
""", unsafe_allow_html=True)
    
with tab_scanner:
    # (Step 1: Upload)
    
    st.markdown("""<div class='step-pill'><span class='step-pill-num'>1</span>Upload Patient DICOM Data</div>""",
                unsafe_allow_html=True)
    
    st.markdown("""
<div style="background:#eff6ff; border:1px solid #dbeafe; border-radius:12px;
                padding:14px 18px; font-size:14px; color:#1e40af; margin-bottom:12px;">
      Select all <strong>.dcm</strong> files from a single patient folder.
      Use <strong>Cmd+A</strong> (Mac) or <strong>Ctrl+A</strong> (Windows) to select all at once.
      At least 20 slice files are required.
</div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload", type=["dcm"], accept_multiple_files=True,
                                      label_visibility="collapsed")
    if uploaded_files:
        st.markdown(f"""
<div style="display:inline-flex; align-items:center; gap:8px;
                    background:#dcfce7; color:#15803d; border-radius:100px;
                    padding:6px 16px; font-size:14px; font-weight:600; margin:6px 0;
                    box-shadow:0 1px 4px rgba(21,128,61,0.15);">
          ✓ &nbsp;{len(uploaded_files)} DICOM files selected
</div>""", unsafe_allow_html=True)
    
    uploaded = uploaded_files or None
    
    
    # (Step 2: Run)
    if uploaded:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("""<div class='step-pill'><span class='step-pill-num'>2</span>Run Analysis</div>""",
                    unsafe_allow_html=True)
    
        cb, ci = st.columns([1, 3])
        with cb:
            run_clicked = st.button("Run Analysis", use_container_width=True)
        with ci:
            lbl = ("Hybrid mode · ResNet-18 CNN false-positive filter active"
                   if mode == "hybrid" else "Classical mode · HU thresholding only")
            st.markdown(f"<div style='color:#64748b;font-size:13px;padding-top:14px;'>{lbl}</div>",
                        unsafe_allow_html=True)
    
        if run_clicked:
            for k in ("results","scored_slices","mode_used","n_slices","viewing_slice"):
                st.session_state.pop(k, None)
    
            with st.spinner("Saving uploaded files…"):
                try:
                    root = save_dcm_files(uploaded_files)
                    if not any(f.lower().endswith(".dcm") for f in os.listdir(root)):
                        st.error("No .dcm files found - please re-select."); st.stop()
                except Exception as e:
                    st.error(f"Save error: {e}"); st.stop()
    
            try:
                from classical.load_ct import load_dicom_series
                with st.spinner("Loading DICOM series…"):
                    series = load_dicom_series(root)
                if len(series) < 10:
                    st.error(f"Only {len(series)} slices - check upload."); st.stop()
            except Exception as e:
                st.error(f"DICOM load error: {e}"); st.stop()
    
            try:
                total, scored_slices = run_pipeline(series, mode, arch)
                st.session_state.update({"results": round(total, 1),
                                         "scored_slices": scored_slices,
                                         "mode_used": mode,
                                         "n_slices": len(series)})
            except Exception as e:
                st.error(f"Pipeline error: {e}"); st.stop()
    
    
    # (Step 3: Results)
    if "results" in st.session_state:
    
        total         = st.session_state["results"]
        scored_slices = st.session_state.get("scored_slices", [])
        mode_used     = st.session_state.get("mode_used", "hybrid")
    
        risk_label, risk_badge, risk_color, risk_bg, risk_border, risk_short, risk_text = get_risk(total)
    
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("""<div class='step-pill'><span class='step-pill-num'>3</span>Results</div>""",
                    unsafe_allow_html=True)
    
        # Score + interpretation
        col_score, col_interp = st.columns([5, 7], gap="large")
    
        with col_score:
            st.markdown(f"""
<div style="background:white; border-radius:16px; padding:32px 24px;
                        text-align:center; border-top:4px solid {risk_color};
                        box-shadow:0 1px 4px rgba(0,0,0,0.06),0 8px 24px rgba(0,0,0,0.06);">
<div style="font-size:11px; font-weight:700; color:#94a3b8;
                          letter-spacing:0.14em; text-transform:uppercase; margin-bottom:10px;">
                Total Agatston Score
</div>
<div style="font-size:5.5rem; font-weight:800; color:{risk_color};
                          line-height:1; margin:4px 0; letter-spacing:-0.03em;">
                {total:.0f}
</div>
<div style="font-size:12px; color:#cbd5e1; margin-bottom:16px; font-weight:500;">
                Agatston units
</div>
<span class="rbadge {risk_badge}">{risk_label}</span>
</div>""", unsafe_allow_html=True)
    
        with col_interp:
            mode_str = "Hybrid · Classical + ResNet-18 CNN" if mode_used == "hybrid" else "Classical · HU Thresholding"
            st.markdown(f"""
<div style="background:{risk_bg}; border:1.5px solid {risk_border}; border-radius:16px;
                        padding:28px; height:100%;
                        box-shadow:0 1px 4px rgba(0,0,0,0.04),0 4px 16px rgba(0,0,0,0.03);">
<div style="font-size:15px; font-weight:700; color:{risk_color}; margin-bottom:10px;">
                {risk_short}
</div>
<div style="font-size:14px; color:#374151; line-height:1.75; margin-bottom:18px;">
                {risk_text}
</div>
<div style="display:flex; gap:20px; padding-top:14px;
                          border-top:1px solid {risk_border}; font-size:12px; color:#94a3b8;">
<span>Pipeline: {mode_str}</span>
<span>·</span>
<span>Slices: {st.session_state.get("n_slices","N/A")}</span>
</div>
</div>""", unsafe_allow_html=True)
    
        if mode_used == "hybrid":
            st.markdown("""
<div style="background:#f0fdf4; border:1px solid #86efac; border-radius:12px;
                        padding:12px 20px; margin-top:14px; font-size:13px; color:#15803d;">
<strong>ResNet-18 CNN filter active</strong> · non-coronary false positives
              were removed before Agatston scoring.
</div>""", unsafe_allow_html=True)
    
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    
        # MESA Percentile
        ref = get_avg_ref(pt_sex, pt_age)
        card_open(border_color="#2563eb")
        section_label("MESA Population Percentile")
    
        if ref is None:
            st.markdown("<div style='color:#64748b;font-size:14px;'>No reference data available for this age range.</div>",
                        unsafe_allow_html=True)
        else:
            pct = score_to_percentile(total, ref)
            if pct < 25:
                pct_color, pct_interp = "#15803d", "This score is **below average** for this age and sex group."
            elif pct < 75:
                pct_color, pct_interp = "#92400e", "This score falls in the **average range** for this age and sex group."
            elif pct < 90:
                pct_color, pct_interp = "#9a3412", "This score is **above the 75th percentile**, indicating elevated cardiovascular risk."
            else:
                pct_color, pct_interp = "#991b1b", "This score is **above the 90th percentile**, indicating high cardiovascular risk."
    
            pa, pb = st.columns([1, 3], gap="large")
            with pa:
                st.markdown(f"""
<div style="text-align:center;">
<div style="font-size:11px; font-weight:700; color:#94a3b8;
                              text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Estimated Percentile
</div>
<div style="font-size:4rem; font-weight:800; color:{pct_color}; line-height:1;">
                    {pct:.0f}<span style="font-size:1.5rem; font-weight:600;">th</span>
</div>
<div style="font-size:12px; color:#94a3b8; margin-top:6px;">
                    {pt_sex}, Age {pt_age}
</div>
</div>""", unsafe_allow_html=True)
            with pb:
                bp = ref["bp"]
                st.markdown(f"""
<div style="font-size:14px; color:#374151; line-height:1.8; padding-top:4px;">
                  A score of <strong style="color:{pct_color};">{total:.0f}</strong> falls at
                  approximately the <strong style="color:{pct_color};">{pct:.0f}th percentile</strong>
                  for {pt_sex.lower()}s aged {pt_age}.<br>
                  {pct_interp}<br>
<span style="font-size:12px; color:#94a3b8;">
                  Reference: 25th pct ≤ {bp.get(25,0)} &nbsp;·&nbsp;
                  50th pct ≤ {bp.get(50,0)} &nbsp;·&nbsp;
                  75th pct ≤ {bp.get(75,0)} &nbsp;·&nbsp;
                  90th pct ≤ {bp.get(90,0)}
</span>
</div>""", unsafe_allow_html=True)
    
            buf = make_percentile_chart(total, pct, pt_sex, pt_age, ref)
            st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
            _, cc, _ = st.columns([0.02, 0.96, 0.02])
            with cc:
                st.image(buf, width=900)
    
        card_close()
    
        # Scoring slices
        if scored_slices:
            card_open()
            section_label("Scoring Slices")
    
            h1, h2, h3 = st.columns([5, 1, 1])
            for col, lbl in zip([h1, h2, h3], ["Filename", "Score", "View"]):
                col.markdown(f"<div style='font-size:11px; font-weight:700; color:#94a3b8; "
                             f"text-transform:uppercase; letter-spacing:0.08em;'>{lbl}</div>",
                             unsafe_allow_html=True)
            st.markdown("<hr style='margin:8px 0; border-color:#f1f5f9; border-width:2px;'>",
                        unsafe_allow_html=True)
    
            for i, (name, hu_arr, score) in enumerate(scored_slices):
                c1, c2, c3 = st.columns([5, 1, 1])
                c1.markdown(f"<div style='font-size:13px;color:#374151;padding:4px 0;'>{name}</div>",
                            unsafe_allow_html=True)
                c2.markdown(f"<div style='font-size:13px;font-weight:700;color:#2563eb;padding:4px 0;'>{score:.1f}</div>",
                            unsafe_allow_html=True)
                with c3:
                    if st.button("👁", key=f"view_{i}", help=f"View CT: {name}"):
                        st.session_state["viewing_slice"] = i
    
            vidx = st.session_state.get("viewing_slice")
            if vidx is not None and 0 <= vidx < len(scored_slices):
                v_name, v_hu, v_score = scored_slices[vidx]
                st.markdown(f"""
<div style="margin-top:16px; padding:14px 18px; background:#f8fafc;
                            border-radius:10px; font-size:13px; color:#64748b;
                            border:1px solid #e2e8f0;">
                  Viewing: <strong style="color:#0f172a;">{v_name}</strong>
                  &nbsp;·&nbsp; Agatston score:
<strong style="color:#2563eb;">{v_score:.1f}</strong>
</div>""", unsafe_allow_html=True)
                _, ic, _ = st.columns([1, 5, 1])
                with ic:
                    st.image(hu_to_uint8(v_hu),
                             caption=f"{v_name}  |  Score: {v_score:.1f}", width=560)
    
            card_close()
        else:
            st.info("No calcified lesions detected in any slice.")
    
        # Disclaimer
        st.markdown("""
<div style="background:white; border:1px solid #e2e8f0; border-radius:12px;
                    padding:14px 20px; font-size:13px; color:#94a3b8;">
<strong style="color:#64748b;">Research Use Only</strong> · This tool is not validated
          for clinical diagnosis or treatment decisions. MESA percentile values are approximate
          population references. Always consult a qualified physician for medical advice.
</div>""", unsafe_allow_html=True)
    