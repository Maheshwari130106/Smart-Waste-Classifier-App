# streamlit_app.py
"""
EcoScan — Final Hybrid UI (Variant C, Option 2 hero) — Updated
- Hero uses local image (rounded card)
- Hero height reduced for mobile
- Top-left icon uses local image (no external links)
- Clickable step pills to jump between steps
- Disposal Help modal (expander)
- Backend: MobileNetV2 (ImageNet) mapping + PDF export (temp files)
"""

import streamlit as st
from io import BytesIO
import tempfile
import os
import numpy as np
from PIL import Image, ImageOps
import requests
from fpdf import FPDF

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image

# -------------------------
# Local hero image path (your uploaded asset)
# -------------------------
HERO_IMAGE_PATH = "assets/hero.png"
# streamlit_app.py
"""
Hybrid Variant C - Smart Waste Classifier
Features:
- Step-by-step wizard (1. Choose Input Method -> 2. Upload/Capture -> 3. See Results)
- Clean green UI (NO background image)
- Upload / Camera / URL options arranged neatly
- Animated transitions (CSS), icons, badges
- Backend: MobileNetV2 (ImageNet) mapping -> waste categories
- PDF report generator
- Cached model loading
- Single-file app
"""

import streamlit as st
from io import BytesIO
import tempfile
import os
import numpy as np
from PIL import Image, ImageOps
import requests
from fpdf import FPDF

# TensorFlow / Keras
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="EcoScan — Waste Classifier (Hybrid UI)",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# CSS (light green background, NO IMAGE)
# ---------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

:root {
  --green-1: #e6f6ec;
  --green-2: #bfead0;
  --accent: #2f855a;
  --accent-2: #16a34a;
  --muted: #64748b;
  --card-bg: rgba(255,255,255,0.9);
}

html, body, [class*="css"] {
  font-family: "Poppins", sans-serif;
}

/* Clean light-green background */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #e8f8ee 0%, #d6f3e3 100%);
}

/* Top hero area */
.hero {
  padding: 22px;
  border-radius: 16px;
  margin-bottom: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.8), rgba(250,255,250,0.8));
  box-shadow: 0 8px 40px rgba(16,24,40,0.06);
}

/* Wizard steps */
.steps {
  display:flex;
  gap:14px;
  align-items:center;
  flex-wrap:wrap;
}
.step {
  padding:10px 18px;
  border-radius:999px;
  background:linear-gradient(90deg, #ffffff, #f2fff6);
  box-shadow: 0 6px 18px rgba(16,24,40,0.04);
  font-weight:600;
  color:#063d16;
  display:flex;
  gap:10px;
  align-items:center;
  transition: transform .18s ease, box-shadow .18s ease;
}
.step.active {
  transform: translateY(-4px);
  box-shadow: 0 18px 40px rgba(23,64,37,0.12);
  background: linear-gradient(90deg, #dff7e9, #c6f0da);
}
.step .dot {
  width:28px; height:28px; border-radius:50%;
  background:linear-gradient(90deg,var(--accent),var(--accent-2));
  color:white; display:flex; align-items:center; justify-content:center; font-weight:700;
}

/* Cards */
.card {
  background: var(--card-bg);
  padding:18px;
  border-radius:14px;
  box-shadow: 0 10px 30px rgba(16,24,40,0.06);
  margin-bottom:18px;
}

/* Input tiles */
.input-tiles {
  display:flex;
  gap:14px;
  flex-wrap:wrap;
}
.tile {
  flex:1 1 220px;
  min-width:180px;
  background: linear-gradient(180deg,#fff,#f7fff8);
  border-radius:12px;
  padding:18px;
  text-align:center;
  cursor:pointer;
  transition: transform .14s ease, box-shadow .14s ease;
  border:1px solid rgba(18,80,50,0.04);
}
.tile:hover{
  transform: translateY(-6px);
  box-shadow: 0 14px 30px rgba(20,70,40,0.06);
}

.small-muted { color:#475569; font-size:13px; }

/* Result badge styles */
.badge {
  padding:8px 14px;
  border-radius:999px;
  color:white; font-weight:700;
  display:inline-block;
}
.badge-plastic { background: linear-gradient(90deg,#ff6b6b,#ff3b30); }
.badge-metal { background: linear-gradient(90deg,#6b7280,#374151); }
.badge-paper { background: linear-gradient(90deg,#3b82f6,#2563eb); }
.badge-cardboard { background: linear-gradient(90deg,#d97706,#a16207); }
.badge-biological { background: linear-gradient(90deg,#10b981,#059669); }
.badge-trash { background: linear-gradient(90deg,#ef4444,#f97316); }
.badge-default { background: linear-gradient(90deg,#9ca3af,#6b7280); }

.preview {
  border-radius:12px;
  overflow:hidden;
  border:1px solid rgba(10,20,10,0.03);
}

@media (max-width: 800px) {
  .input-tiles { flex-direction:column; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Hero section
# ---------------------------
st.markdown('<div class="hero card">', unsafe_allow_html=True)
colh1, colh2 = st.columns([3, 1])
with colh1:
    st.markdown("<h2 style='margin:0'>♻ EcoScan — Smart Waste Classifier</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Step-by-step wizard to upload/capture & classify waste instantly.</div>", unsafe_allow_html=True)
with colh2:
    st.markdown("<div class='center'><img src='https://img.icons8.com/fluency/48/000000/recycle.png'></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# ---------------------------
# Waste map & metadata
# ---------------------------
waste_map = {
    "battery": ["battery","accumulator","cell"],
    "biological": [
        "banana","apple","vegetable","fruit","leaf","leaves","branch","twig","plant","flower"
    ],
    "cardboard": ["carton","cardboard","box"],
    "clothes": ["t-shirt","jeans","sweater","coat","cloth"],
    "metal": ["can","tin","aluminum","metal"],
    "paper": ["newspaper","envelope","paper","book_jacket","notebook"],
    "plastic": ["plastic","plastic_bag","water_bottle","packet","bottlecap","container"],
    "shoes": ["shoe","sneaker","boot","sandal"],
    "trash": ["garbage","trash_can","waste"],
    "white-glass": ["glass","clear_glass","glassware"]
}

recyclability = {
    "battery":20,"biological":95,"cardboard":92,"clothes":60,"metal":98,
    "paper":88,"plastic":55,"shoes":40,"trash":10,"white-glass":90
}

instructions = {
    "battery":"Do not throw in trash. Use battery recycling bins.",
    "biological":"Compost or dispose in organic waste.",
    "cardboard":"Flatten and keep dry before recycling.",
    "clothes":"Donate usable clothes or recycle textiles.",
    "metal":"Rinse and recycle metal cans.",
    "paper":"Recycle clean and dry paper.",
    "plastic":"Rinse and follow local recycling rules.",
    "shoes":"Donate or recycle via shoe programs.",
    "trash":"Dispose in general waste.",
    "white-glass":"Rinse and recycle clear glass."
}

carbon_score = {
    "battery":5,"biological":1,"cardboard":2,"clothes":3,"metal":4,
    "paper":2,"plastic":5,"shoes":3,"trash":4,"white-glass":3
}

# ---------------------------
# Helpers
# ---------------------------
def pil_from_upload(u): return Image.open(u).convert("RGB")

def load_image_from_url(url):
    r = requests.get(url, timeout=10); r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def preprocess_pil(img, size=(224,224)):
    img_r = img.resize(size)
    arr = image.img_to_array(img_r)
    x = np.expand_dims(arr,0)
    x = preprocess_input(x)
    return x, arr

def map_imagenet_to_waste(label):
    label = label.lower()
    for w,keys in waste_map.items():
        if any(k in label for k in keys):
            return w
    return "trash"

def estimate_quantity(arr):
    pil = Image.fromarray(arr.astype('uint8'))
    gray = ImageOps.grayscale(pil).resize((224,224))
    a = np.array(gray).astype(float)
    mask = a < (np.median(a)*0.85)
    p = mask.sum()/mask.size
    return "Large" if p>0.30 else "Medium" if p>0.12 else "Small"

def compute_recyclability_score(w,c):
    base = recyclability.get(w,50)
    return max(0,min(100,int(base*(0.5+0.5*(c/100)))))

def compute_carbon(w):
    lvl = carbon_score.get(w,3)
    return lvl, round(lvl*2.0,2)

def create_pdf_report(report, image_pil):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10,"EcoScan - Waste Report",ln=True,align="C")
    pdf.set_font("Arial", size=10)
    for k,v in report.items():
        pdf.multi_cell(0,8,f"{k}: {v}")
    if image_pil:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".png") as tmp:
            image_pil.save(tmp.name,"PNG")
            pdf.image(tmp.name,w=120)
            os.unlink(tmp.name)
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        data=open(tmp.name,"rb").read()
        os.unlink(tmp.name)
    return data

# ---------------------------
# Wizard State
# ---------------------------
if "step" not in st.session_state: st.session_state.step=1
if "chosen_method" not in st.session_state: st.session_state.chosen_method=None
if "uploaded_image" not in st.session_state: st.session_state.uploaded_image=None
if "image_url" not in st.session_state: st.session_state.image_url=""
if "camera_image" not in st.session_state: st.session_state.camera_image=None

# ---------------------------
# Step 1 – Input Method
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="steps">', unsafe_allow_html=True)

def _step_html(n,l,active=False):
    return f'<div class="step {"active" if active else ""}"><div class="dot">{n}</div>{l}</div>'

st.markdown(_step_html(1,"Choose Input",st.session_state.step==1),unsafe_allow_html=True)
st.markdown(_step_html(2,"Upload / Capture",st.session_state.step==2),unsafe_allow_html=True)
st.markdown(_step_html(3,"Results",st.session_state.step==3),unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<h4>Step 1 — Choose input method</h4>",unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)
with col1:
    if st.button("📁 Upload Image"): 
        st.session_state.step=2; st.session_state.chosen_method="upload"
with col2:
    if st.button("📷 Use Camera"):
        st.session_state.step=2; st.session_state.chosen_method="camera"
with col3:
    if st.button("🔗 Paste Image URL"):
        st.session_state.step=2; st.session_state.chosen_method="url"

st.markdown("</div>",unsafe_allow_html=True)

# ---------------------------
# Step 2 – Upload / Capture
# ---------------------------
if st.session_state.step==2:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.markdown("<h4>Step 2 — Upload / Capture</h4>",unsafe_allow_html=True)

    if st.session_state.chosen_method=="upload":
        img = st.file_uploader("Choose image (jpg/png)",type=["jpg","jpeg","png"])
        if img:
            st.session_state.uploaded_image = pil_from_upload(img)
            if st.button("➡ Go to Results"):
                st.session_state.step=3

    elif st.session_state.chosen_method=="camera":
        cam = st.camera_input("Take a photo")
        if cam:
            st.session_state.camera_image = Image.open(cam).convert("RGB")
            if st.button("➡ Go to Results"):
                st.session_state.step=3

    elif st.session_state.chosen_method=="url":
        url = st.text_input("Enter image URL", value=st.session_state.image_url)
        if st.button("Load from URL"):
            try:
                st.session_state.uploaded_image = load_image_from_url(url)
                st.session_state.image_url = url
                st.success("Loaded successfully.")
            except:
                st.error("Invalid URL.")
        if st.button("➡ Go to Results"):
            if st.session_state.uploaded_image is not None:
                st.session_state.step=3
            else:
                st.warning("Please load image.")

    if st.button("⬅ Back"):
        st.session_state.step=1; st.session_state.chosen_method=None

    st.markdown("</div>",unsafe_allow_html=True)

# ---------------------------
# Step 3 – Results
# ---------------------------
if st.session_state.step==3:

    img = st.session_state.camera_image or st.session_state.uploaded_image
    if img is None:
        st.error("No image. Go back.")
    else:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown("<h4>Step 3 — Results</h4>",unsafe_allow_html=True)

        lc, rc = st.columns([1.2,1])

        with lc:
            st.markdown("<div class='preview'>",unsafe_allow_html=True)
            st.image(img,use_column_width=True)
            st.markdown("</div>",unsafe_allow_html=True)

        x, arr = preprocess_pil(img)
        preds = model.predict(x)
        decoded = decode_predictions(preds,top=3)[0]

        top_list=[]
        for p in decoded:
            lbl = p[1]
            conf = p[2]*100
            mapped = map_imagenet_to_waste(lbl)
            top_list.append({"imagenet_label":lbl,"confidence":conf,"mapped":mapped})

        final = top_list[0]
        waste = final["mapped"]
        conf = final["confidence"]
        qty = estimate_quantity(arr)
        recy = compute_recyclability_score(waste,conf)
        lvl, co2 = compute_carbon(waste)
        disp = instructions.get(waste,"Follow rules.")

        with rc:
            st.markdown(f"<div class='badge badge-{waste.replace('-','_')}' style='font-size:18px'>{waste.upper()}</div>",unsafe_allow_html=True)
            st.markdown(f"**Top Label:** `{final['imagenet_label']}`")
            st.markdown(f"**Confidence:** {conf:.2f}%")
            st.markdown(f"**Estimated Quantity:** {qty}")
            st.markdown(f"**Recyclability Score:** {recy}/100")
            st.progress(recy/100)
            st.markdown(f"**Carbon Level:** {lvl} (~{co2} kg CO₂)")
            st.success(disp)

        st.markdown("---")
        st.subheader("Top Predictions")
        for i,t in enumerate(top_list,1):
            st.write(f"{i}. `{t['imagenet_label']}` — {t['confidence']:.2f}% → **{t['mapped']}**")

        report = {
            "Waste Type": waste,
            "Top Label": final['imagenet_label'],
            "Confidence (%)": f"{conf:.2f}",
            "Quantity Estimate": qty,
            "Recyclability": f"{recy}/100",
            "Carbon": f"{lvl} (~{co2}kg CO2)",
            "Instructions": disp,
        }

        pdf = create_pdf_report(report,img)

        colA,colB = st.columns(2)
        with colA:
            if st.button("🔁 Analyze New Image"):
                st.session_state.step=1
                st.session_state.uploaded_image=None
                st.session_state.camera_image=None
                st.session_state.image_url=""
                st.session_state.chosen_method=None
                st.experimental_rerun()

        with colB:
            st.download_button("📄 Download PDF",data=pdf,file_name="ecoscan_report.pdf",mime="application/pdf")

        st.markdown("</div>",unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    "<div style='text-align:center; margin-top:18px; color:#475569; font-size:13px'>"
    "Model uses MobileNetV2 (ImageNet). For production, use YOLO/DETR fine-tuned on waste datasets."
    "</div>",
    unsafe_allow_html=True,
)

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="EcoScan — Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------
# CSS — Rounded hero banner on Step 1, pastel-green theme for other steps
# -------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

:root {{
  --pastel-green: #EAFBE8;
  --accent-dark: #19692f;
  --accent: #2f855a;
  --muted: #475569;
  --card-bg: #ffffff;
}}

html, body, [class*="css"] {{
  font-family: "Poppins", sans-serif;
  background: var(--pastel-green);
}}

/* Top bar */
.topbar {{
  display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;
}}
.topbar-left {{ display:flex; align-items:center; gap:12px }}

/* Hero rounded card only used in Step 1 */
.hero-card {{
  width: 100%;
  border-radius: 20px;
  overflow: hidden;
  display: flex;
  gap: 18px;
  align-items: center;
  justify-content: space-between;
  padding: 0;
  box-shadow: 0 12px 40px rgba(16,24,40,0.06);
  margin-bottom: 18px;
  background: white;
}}

.hero-left {{
  padding: 28px;
  min-width: 48%;
}}
.hero-right {{
  width: 48%;
  min-height: 260px;
  background-image: url("file://{HERO_IMAGE_PATH}");
  background-size: cover;
  background-position: center;
}}

/* make hero smaller on mobile */
@media (max-width: 900px) {{
  .hero-card {{ flex-direction:column; gap:0; }}
  .hero-left, .hero-right {{ width:100%; }}
  .hero-right {{ min-height:160px; }}
  .hero-left {{ padding:18px; }}
}}

/* small text */
.lead {{
  color: #0f172a;
  font-weight: 700;
  font-size: 28px;
  margin-bottom: 8px;
}}
.lead-sub {{
  color: #254a35;
  font-weight: 500;
  margin-bottom: 14px;
}}

.cta {{
  background: linear-gradient(90deg,var(--accent),#16a34a);
  color:white;
  border-radius:12px;
  padding:10px 16px;
  font-weight:700;
  border:none;
}

/* Steps indicator */
.steps {{
  display:flex;
  gap:12px;
  align-items:center;
  margin-bottom:14px;
  flex-wrap:wrap;
}}
.step {{
  background: #fff;
  padding:8px 12px;
  border-radius:999px;
  color:var(--accent-dark);
  font-weight:600;
  box-shadow: 0 6px 18px rgba(16,24,40,0.04);
  cursor:pointer;
  display:flex;
  gap:8px;
  align-items:center;
}}
.step.active {{
  background: linear-gradient(90deg,#e8f9ee,#d6f3df);
  transform: translateY(-4px);
  box-shadow: 0 14px 30px rgba(23,64,37,0.08);
}}

/* Cards */
.card {{
  background: var(--card-bg);
  padding:18px;
  border-radius:12px;
  box-shadow: 0 8px 30px rgba(16,24,40,0.04);
  margin-bottom:16px;
}}

/* Input tiles */
.input-tiles {{
  display:flex;
  gap:14px;
  flex-wrap:wrap;
}}
.tile {{
  flex:1 1 220px;
  min-width:180px;
  background: linear-gradient(180deg,#ffffff,#f7fff8);
  border-radius:12px;
  padding:16px;
  text-align:center;
  cursor:pointer;
  transition: transform .12s ease, box-shadow .12s ease;
  border:1px solid rgba(18,80,50,0.04);
}}
.tile:hover{{ transform: translateY(-6px); box-shadow: 0 14px 30px rgba(20,70,40,0.06); }}
.tile .icon {{
  width:54px; height:54px; border-radius:12px; display:inline-flex; align-items:center; justify-content:center;
  margin-bottom:10px; font-size:26px;
  color:white;
}}
.icon.upload {{ background: linear-gradient(90deg,#34d399,#059669); }}
.icon.camera {{ background: linear-gradient(90deg,#60a5fa,#3b82f6); }}
.icon.url {{ background: linear-gradient(90deg,#f59e0b,#f97316); }}

.small-muted {{ color:var(--muted); font-size:13px; }}

/* preview image */
.preview {{
  border-radius:10px;
  overflow:hidden;
  border:1px solid rgba(10,20,10,0.03);
}}

/* badges */
.badge {{
  padding:8px 14px;
  border-radius:999px;
  color:white; font-weight:700; display:inline-block;
}}
.badge-plastic {{ background: linear-gradient(90deg,#ff6b6b,#ff3b30); }}
.badge-biological {{ background: linear-gradient(90deg,#10b981,#059669); }}
.badge-paper {{ background: linear-gradient(90deg,#3b82f6,#2563eb); }}
.badge-metal {{ background: linear-gradient(90deg,#6b7280,#374151); }}
.badge-trash {{ background: linear-gradient(90deg,#ef4444,#f97316); }}
.badge-default {{ background: linear-gradient(90deg,#9ca3af,#6b7280); }}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Top bar: use local image as icon (no external links)
# -------------------------
col_t1, col_t2 = st.columns([0.2, 0.8])
with col_t1:
    # show a small thumbnail of the hero image as icon
    try:
        st.image(HERO_IMAGE_PATH, width=48)
    except Exception:
        # fallback to emoji
        st.markdown("♻️")
with col_t2:
    st.markdown("<div style='font-weight:700; color:#064e3b; font-size:18px'>EcoScan</div>", unsafe_allow_html=True)

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# -------------------------
# Mapping + metadata
# -------------------------
waste_map = {
    "battery": ["battery", "accumulator", "power_cell", "cell"],
    "biological": [
        "banana", "apple", "vegetable", "fruit", "food", "meat", "organic",
        "leaf", "leaves", "branch", "twig", "plant", "shrubs", "foliage",
        "tree", "grass", "bush", "flower", "petal"
    ],
    "brown-glass": ["beer_bottle", "wine_bottle"],
    "cardboard": ["carton", "cardboard", "box", "pizza_box"],
    "clothes": ["tshirt","cloth","jeans","sweater","coat"],
    "green-glass": ["bottle_green","wine_bottle"],
    "metal": ["can","tin","aluminum","screw","metal"],
    "paper": ["newspaper","envelope","paper","book_jacket","notebook"],
    "plastic": ["plastic","plastic_bag","water_bottle","packet","bottlecap","container"],
    "shoes": ["shoe","sneaker","boot","sandal"],
    "trash": ["garbage","trash_can","waste"],
    "white-glass": ["glass","clear_glass","glassware","bottle_white"]
}

recyclability = {
    "battery": 20, "biological": 95, "brown-glass": 90, "cardboard": 92, "clothes": 60,
    "green-glass": 90, "metal": 98, "paper": 88, "plastic": 55, "shoes": 40, "trash": 10, "white-glass": 90
}

instructions = {
    "battery": "Do not throw in general trash. Take to battery recycling bins.",
    "biological": "Compost or put in wet/organic waste. Do not mix with recyclables.",
    "brown-glass": "Rinse and recycle with brown glass.",
    "cardboard": "Flatten and place in dry recycling.",
    "clothes": "Donate usable clothing; otherwise recycle as textile waste.",
    "green-glass": "Rinse and recycle with green glass.",
    "metal": "Rinse and send for metal recycling.",
    "paper": "Keep dry and place in paper recycling stream.",
    "plastic": "Rinse containers & follow local recycling rules.",
    "shoes": "Donate if wearable; else check shoe recycling programs.",
    "trash": "Dispose in general waste; separate recyclables if possible.",
    "white-glass": "Recycle with clear/white glass."
}

carbon_score = {
    "battery": 5, "biological": 1, "brown-glass": 3, "cardboard": 2, "clothes": 3,
    "green-glass": 3, "metal": 4, "paper": 2, "plastic": 5, "shoes": 3, "trash": 4, "white-glass": 3
}

# -------------------------
# Helpers
# -------------------------
def pil_from_upload(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")

def load_image_from_url(url: str):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def preprocess_pil(img: Image.Image, size=(224,224)):
    img_resized = img.resize(size)
    arr = image.img_to_array(img_resized)
    x = np.expand_dims(arr, axis=0)
    x = preprocess_input(x)
    return x, arr

def map_imagenet_to_waste(label: str):
    lab = label.lower()
    for wtype, keys in waste_map.items():
        for k in keys:
            if k in lab:
                return wtype
    if "bottle" in lab or "jar" in lab:
        return "plastic" if "plastic" in lab or "water" in lab else "brown-glass"
    if "paper" in lab or "envelope" in lab:
        return "paper"
    return "trash"

def estimate_quantity(arr):
    pil = Image.fromarray(arr.astype('uint8'))
    gray = ImageOps.grayscale(pil).resize((224,224))
    arrg = np.array(gray).astype(np.float32)
    th = np.median(arrg) * 0.85
    mask = arrg < th
    proportion = mask.sum() / mask.size
    if proportion > 0.30:
        return "Large (high visible waste area)"
    elif proportion > 0.12:
        return "Medium (moderate visible waste area)"
    else:
        return "Small (small visible waste area)"

def compute_recyclability_score(waste_type, confidence):
    base = recyclability.get(waste_type, 50)
    adj = int(base * (0.5 + 0.5 * (confidence/100.0)))
    return max(0, min(100, adj))

def compute_carbon(waste_type):
    lvl = carbon_score.get(waste_type, 3)
    approx = round(lvl * 2.0, 2)
    return lvl, approx

# -------------------------
# PDF creation
# -------------------------
def create_pdf_report(report_dict, image_pil):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "EcoScan - Waste Classification Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    for k, v in report_dict.items():
        pdf.multi_cell(0, 8, f"{k}: {v}")
    pdf.ln(6)
    if image_pil:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            image_pil.save(tmp_img.name, format="PNG")
            pdf.image(tmp_img.name, w=120)
            try:
                os.unlink(tmp_img.name)
            except:
                pass
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            data = f.read()
        try:
            os.unlink(tmp_pdf.name)
        except:
            pass
    return data

# -------------------------
# Session-state wizard
# -------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "chosen_method" not in st.session_state:
    st.session_state.chosen_method = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "camera_image" not in st.session_state:
    st.session_state.camera_image = None
if "image_url" not in st.session_state:
    st.session_state.image_url = ""

def set_step(step):
    st.session_state.step = step

def choose_method(method):
    st.session_state.chosen_method = method
    st.session_state.step = 2

# -------------------------
# Step indicator (clickable pills)
# -------------------------
st.markdown('<div class="steps">', unsafe_allow_html=True)
# step 1 pill
if st.button("1 • Choose Input", key="pill1"):
    set_step(1)
st.markdown('<div style="width:8px"></div>', unsafe_allow_html=True)
# step2 pill
if st.button("2 • Upload / Capture", key="pill2"):
    set_step(2)
# step3 pill
if st.button("3 • Results", key="pill3"):
    set_step(3)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Step 1: Hero Banner (rounded) — only when step==1
# -------------------------
if st.session_state.step == 1:
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<div class="hero-left">', unsafe_allow_html=True)
    st.markdown('<div class="lead">Platform for Waste Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="lead-sub">Make a photo or upload an image and EcoScan will determine the type of garbage and suggest disposal steps.</div>', unsafe_allow_html=True)
    # CTA to proceed selects input choice view (keeps user on step 1 but scrolls to tiles)
    if st.button("Get Started"):
        st.session_state.step = 1
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-right"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Input tiles (always visible below hero)
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h4 style='margin-top:6px'>Step 1 — Choose Input Method</h4>", unsafe_allow_html=True)
st.markdown("<div class='input-tiles'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📁 Upload Image", key="btn_upload_tile"):
        choose_method("upload")
    st.caption("Upload from device")
with col2:
    if st.button("📷 Use Camera", key="btn_camera_tile"):
        choose_method("camera")
    st.caption("Take photo with your device camera")
with col3:
    if st.button("🔗 Paste Image URL", key="btn_url_tile"):
        choose_method("url")
    st.caption("Load image from web link")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Step 2 — Upload / Camera / URL
# -------------------------
if st.session_state.step == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4>Step 2 — Upload / Capture</h4>", unsafe_allow_html=True)

    if st.session_state.chosen_method == "upload":
        uploaded = st.file_uploader("Choose an image file (jpg / png)", type=["jpg","jpeg","png"], key="upfile")
        if uploaded:
            st.session_state.uploaded_image = pil_from_upload(uploaded)
            st.success("Image uploaded — press 'Go to Results'")
    elif st.session_state.chosen_method == "camera":
        cam = st.camera_input("Take a photo (camera)", key="cam_input")
        if cam:
            st.session_state.camera_image = Image.open(cam).convert("RGB")
            st.success("Photo captured — press 'Go to Results'")
    elif st.session_state.chosen_method == "url":
        url = st.text_input("Paste image URL here", value=st.session_state.image_url, key="url_input")
        if st.button("Load from URL", key="btn_load_url"):
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                st.session_state.uploaded_image = Image.open(BytesIO(r.content)).convert("RGB")
                st.session_state.image_url = url
                st.success("Image loaded from URL")
            except Exception as e:
                st.error(f"Could not load image: {e}")

    st.markdown("<div style='display:flex; gap:8px; margin-top:12px'>", unsafe_allow_html=True)
    if st.button("➡️ Go to Results", key="to_results"):
        if st.session_state.camera_image or st.session_state.uploaded_image:
            st.session_state.step = 3
        else:
            st.warning("Please upload, capture, or load an image first.")
    if st.button("⬅️ Back", key="back_to_step1"):
        st.session_state.step = 1
        st.session_state.chosen_method = None
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Disposal Help modal (expander) — accessible anywhere
# -------------------------
with st.expander("❓ Disposal Help (how to dispose each category)", expanded=False):
    st.markdown("**Battery:** Take to dedicated battery recycling points — do not throw in regular trash.")
    st.markdown("**Biological / Organic:** Compost or place in wet waste. Keep food/vegetable waste separate from dry recyclables.")
    st.markdown("**Paper / Cardboard:** Keep dry; flatten boxes; drop in paper recycling.")
    st.markdown("**Glass (brown/green/white):** Rinse and place in correct glass stream. Do not mix glass colors if local rules require separation.")
    st.markdown("**Plastic:** Rinse containers when possible; follow local plastic codes (PET, HDPE separation).")
    st.markdown("**Metal:** Rinse and send to metal recycling or scrap collectors.")
    st.markdown("**Clothes / Shoes:** Donate usable items; otherwise use textile recycling programs.")
    st.markdown("**Trash:** Non-recyclable items — dispose as general waste.")

# -------------------------
# Step 3 — Results
# -------------------------
if st.session_state.step == 3:
    image_pil = st.session_state.camera_image or st.session_state.uploaded_image

    if image_pil is None:
        st.error("No image found. Go back and upload or capture an image.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>Step 3 — Results</h4>", unsafe_allow_html=True)
        left, right = st.columns([1.3, 1])

        with left:
            st.markdown('<div class="preview">', unsafe_allow_html=True)
            st.image(image_pil, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # predict
        x, arr = preprocess_pil(image_pil, size=(224,224))
        with st.spinner("Running model..."):
            preds = model.predict(x)
        decoded = decode_predictions(preds, top=3)[0]

        top_list = []
        for p in decoded:
            lbl = p[1]
            conf = float(p[2]) * 100.0
            mapped = map_imagenet_to_waste(lbl)
            top_list.append({"imagenet_label": lbl, "confidence": conf, "mapped": mapped})

        final = top_list[0]
        waste_type = final["mapped"]
        confidence = final["confidence"]
        quantity = estimate_quantity(arr)
        recy = compute_recyclability_score(waste_type, confidence)
        carbon_lvl, carbon_kg = compute_carbon(waste_type)
        disposal = instructions.get(waste_type, "Follow local disposal guidelines.")

        with right:
            badge_cls = waste_type.replace("-", "_")
            st.markdown(f"<div class='badge badge-{badge_cls}' style='font-size:18px'>{waste_type.upper()}</div>", unsafe_allow_html=True)
            st.markdown(f"**Top ImageNet label:** `{final['imagenet_label']}`")
            st.markdown(f"**Confidence:** **{confidence:.2f}%**")
            st.markdown(f"**Estimated Quantity:** {quantity}")
            st.markdown(f"**Recyclability Score:** {recy}/100")
            st.progress(min(100, recy)/100.0)
            st.markdown(f"**Carbon Footprint:** Level {carbon_lvl} / 5 (~{carbon_kg} kg CO₂)")
            st.markdown("**Disposal Instructions:**")
            st.success(disposal)

        st.markdown("---")
        st.subheader("Top Predictions")
        for i, t in enumerate(top_list, start=1):
            st.write(f"{i}. `{t['imagenet_label']}` — {t['confidence']:.2f}% → **{t['mapped']}**")

        # PDF
        report = {
            "Waste Type": waste_type,
            "Top Label": final['imagenet_label'],
            "Confidence (%)": f"{confidence:.2f}",
            "Estimated Quantity": quantity,
            "Recyclability Score": f"{recy}/100",
            "Carbon (level/kgCO2)": f"{carbon_lvl} / {carbon_kg}",
            "Disposal Instructions": disposal,
            "All Predictions": "; ".join([f"{k['imagenet_label']} ({k['confidence']:.1f}%)->{k['mapped']}" for k in top_list])
        }

        pdf_bytes = create_pdf_report(report, image_pil)

        st.markdown("<div style='display:flex; gap:10px; margin-top:12px'>", unsafe_allow_html=True)
        if st.button("🔁 Analyze New Image"):
            st.session_state.step = 1
            st.session_state.chosen_method = None
            st.session_state.uploaded_image = None
            st.session_state.camera_image = None
            st.session_state.image_url = ""
            st.experimental_rerun()
        st.download_button("📄 Download Report (PDF)", data=pdf_bytes, file_name="ecoscan_report.pdf", mime="application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer note
# -------------------------
st.markdown(
    "<div style='text-align:center; color:#254a35; margin-top:14px; font-size:13px'>"
    "Note: Uses MobileNetV2 (ImageNet). For production-level accuracy consider fine-tuning on labeled waste dataset or using object detection (YOLOv8/DETR)."
    "</div>",
    unsafe_allow_html=True
)
