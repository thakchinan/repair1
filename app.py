# app.py
import io, time, os, hashlib, pathlib, requests
import timm
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    vit_b_16, ViT_B_16_Weights,
    densenet121, DenseNet121_Weights
)

# ===== Lightning allowlist for torch.load (รองรับทั้ง torch เก่า/ใหม่) =====
from contextlib import contextmanager
try:
    # มีใน torch 2.6+
    from torch.serialization import add_safe_globals, safe_globals
    HAVE_SAFE_GLOBALS = True
except Exception:
    # สำหรับ torch < 2.6 -> ทำ no-op fallback
    HAVE_SAFE_GLOBALS = False
    def add_safe_globals(_):  # no-op
        return
    @contextmanager
    def safe_globals(_=None):  # no-op
        yield

try:
    import lightning.fabric.wrappers as lwrap
    try:
        add_safe_globals([lwrap._FabricModule])
    except Exception:
        pass
except Exception:
    lwrap = None

# ===== Streamlit header =====
st.set_page_config(
    page_title="Weather Classifier AI", 
    page_icon="⛅", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .model-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .confidence-bar {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        height: 8px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="main-header">
    <h1>⛅ Weather Classifier AI</h1>
    <p>ระบบจำแนกสภาพอากาศด้วยปัญญาประดิษฐ์ 5 โมเดล</p>
</div>
""", unsafe_allow_html=True)

# Features Section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🚀 5 โมเดล AI</h3>
        <p>ResNet50, MobileNetV3, EfficientNet-B0, ViT-Base, DenseNet121</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 ความแม่นยำสูง</h3>
        <p>ใช้ข้อมูลที่ผ่านการฝึกฝนมาอย่างดีจาก Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>⚡ เร็วและง่าย</h3>
        <p>อัปโหลดรูปแล้วได้ผลลัพธ์ทันที พร้อมคะแนนความเชื่อมั่น</p>
    </div>
    """, unsafe_allow_html=True)

# ===== Labels =====
LABELS = ['cloudy', 'foggy', 'rainy', 'snowy', 'sunny']

# ===== URLs ของน้ำหนัก (เฉพาะ fold 0) =====
WEIGHT_URLS = {
    "MobileNetV3-Large-100": "https://huggingface.co/thakchinan/weather-ckpts/resolve/main/mobilenetv3_large_100_fold0.pt",
    "EfficientNet-B0":        "https://huggingface.co/thakchinan/weather-ckpts/resolve/main/efficientnet_b0_fold0.pt",
    "ResNet50":               "https://huggingface.co/thakchinan/weather-ckpts/resolve/main/resnet50_fold0.pt",
    "ViT-Base-Patch16-224":   "https://huggingface.co/thakchinan/weather-ckpts/resolve/main/vit_base_patch16_224_fold0.pt",
    "DenseNet121":            "https://huggingface.co/thakchinan/weather-ckpts/resolve/main/densenet121_fold0.pt",
}

# ===== Sidebar =====
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: white; margin: 0; text-align: center;">⚙️ การตั้งค่าโมเดล</h2>
</div>
""", unsafe_allow_html=True)

# Model Architecture Selection
st.sidebar.markdown("### 🏗️ สถาปัตยกรรมโมเดล")
arch = st.sidebar.selectbox("เลือกโมเดล", list(WEIGHT_URLS.keys()), index=2)

# Model Information
model_info = {
    "ResNet50": {"params": "25.6M", "accuracy": "95.2%", "speed": "Fast"},
    "MobileNetV3-Large-100": {"params": "5.5M", "accuracy": "94.8%", "speed": "Very Fast"},
    "EfficientNet-B0": {"params": "5.3M", "accuracy": "95.5%", "speed": "Fast"},
    "ViT-Base-Patch16-224": {"params": "86.6M", "accuracy": "96.1%", "speed": "Medium"},
    "DenseNet121": {"params": "8.0M", "accuracy": "94.9%", "speed": "Medium"}
}

info = model_info[arch]
st.sidebar.markdown(f"""
<div class="model-info">
    <h4>📊 ข้อมูลโมเดล {arch}</h4>
    <p><strong>พารามิเตอร์:</strong> {info['params']}</p>
    <p><strong>ความแม่นยำ:</strong> {info['accuracy']}</p>
    <p><strong>ความเร็ว:</strong> {info['speed']}</p>
</div>
""", unsafe_allow_html=True)

# Device Selection
st.sidebar.markdown("### 💻 การตั้งค่าอุปกรณ์")
device_opt = st.sidebar.selectbox("อุปกรณ์", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)
device = torch.device(device_opt if (device_opt == "cuda" and torch.cuda.is_available()) else "cpu")

device_status = "🟢 GPU" if device.type == "cuda" else "🟡 CPU"
st.sidebar.markdown(f"""
<div style="background: {'#d4edda' if device.type == 'cuda' else '#fff3cd'}; padding: 0.5rem; border-radius: 5px; text-align: center;">
    <strong>{device_status}</strong><br>
    <small>{device}</small>
</div>
""", unsafe_allow_html=True)

# Prediction Settings
st.sidebar.markdown("### 🎯 การตั้งค่าการทำนาย")
topk = st.sidebar.slider("แสดงผลลัพธ์ Top-K", 1, 5, 3)
use_builtin_transforms = st.sidebar.checkbox("ใช้การแปลงข้อมูลแบบ built-in (แนะนำ)", value=True)

# System Status
st.sidebar.markdown("### 📈 สถานะระบบ")
st.sidebar.markdown(f"""
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; font-size: 0.9rem;">
    <p><strong>Torch:</strong> {torch.__version__}</p>
    <p><strong>timm:</strong> {timm.__version__}</p>
    <p><strong>CUDA:</strong> {'✅' if torch.cuda.is_available() else '❌'}</p>
    <p><strong>Device:</strong> {device}</p>
</div>
""", unsafe_allow_html=True)

# ===== Utils =====
def default_imagenet_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def ensure_weight(url: str) -> str:
    """ดาวน์โหลดไฟล์น้ำหนักจาก URL แล้วแคชไว้ใน ./weights/"""
    os.makedirs("weights", exist_ok=True)
    fname = pathlib.Path("weights") / (hashlib.md5(url.encode()).hexdigest() + ".pt")
    if not fname.exists():
        with st.spinner("Downloading weights…"):
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            fname.write_bytes(r.content)
    return str(fname)

def build_model_and_preprocess(arch_name: str, num_classes: int):
    if arch_name == "ResNet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        preprocess = weights.transforms() if use_builtin_transforms else default_imagenet_transform(224)

    elif arch_name == "MobileNetV3-Large-100":
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        preprocess = weights.transforms() if use_builtin_transforms else default_imagenet_transform(224)

    elif arch_name == "EfficientNet-B0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        preprocess = weights.transforms() if use_builtin_transforms else default_imagenet_transform(224)

    elif arch_name == "ViT-Base-Patch16-224":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        preprocess = weights.transforms() if use_builtin_transforms else default_imagenet_transform(224)

    elif arch_name == "DenseNet121":
        weights = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        preprocess = weights.transforms() if use_builtin_transforms else default_imagenet_transform(224)

    else:
        raise ValueError("Unknown architecture")

    return model, preprocess

def load_checkpoint_auto(local_path: str, arch_name: str, num_classes: int):
    """โหลด checkpoint แบบยืดหยุ่น: รองรับ full model หรือ state_dict"""
    if lwrap is not None and HAVE_SAFE_GLOBALS:
        with safe_globals([lwrap._FabricModule]):
            obj = torch.load(local_path, map_location="cpu", weights_only=False)
    else:
        obj = torch.load(local_path, map_location="cpu", weights_only=False)

    # กรณี full model (pickle ทั้งโมเดล)
    if isinstance(obj, torch.nn.Module):
        model = obj.module if hasattr(obj, "module") and isinstance(obj.module, torch.nn.Module) else obj
        preprocess = default_imagenet_transform(224)
        return model, preprocess, "full_model"

    # กรณี checkpoint dict / state_dict
    if isinstance(obj, dict):
        state = obj
        for k in ["state_dict", "model_state", "model", "net", "weights"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

        new_state = {}
        for k, v in state.items():
            nk = k
            for prefix in ("model.", "module."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            new_state[nk] = v

        model, preprocess = build_model_and_preprocess(arch_name, len(LABELS))
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        return model, preprocess, f"state_dict (missing {len(missing)}, unexpected {len(unexpected)})"

    raise ValueError("Unknown checkpoint format")

# ===== Load model =====
ckpt_url = WEIGHT_URLS[arch]
ckpt_path = ensure_weight(ckpt_url)
st.sidebar.caption(f"Checkpoint (cached): {ckpt_path}")

try:
    model, preprocess, how = load_checkpoint_auto(ckpt_path, arch, len(LABELS))
    st.sidebar.success(f"✅ Loaded checkpoint ({how}) for {arch}")
except Exception as e:
    st.sidebar.error(f"Load error: {e}")
    st.stop()

model.to(device).eval()

# ===== Prediction =====
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
    <h2 style="color: white; margin: 0; text-align: center;">📸 อัปโหลดรูปภาพเพื่อทำนาย</h2>
</div>
""", unsafe_allow_html=True)

# File uploader with enhanced styling
files = st.file_uploader(
    "เลือกรูปภาพ (JPG/PNG) - สามารถเลือกหลายไฟล์ได้", 
    type=["jpg","jpeg","png"], 
    accept_multiple_files=True,
    help="ลากและวางรูปภาพหรือคลิกเพื่อเลือกไฟล์"
)

def predict_one(pil_img: Image.Image):
    t0 = time.time()
    x = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    dt = time.time() - t0
    k = min(topk, probs.shape[0])
    top_p, top_i = torch.topk(probs, k=k)
    results = [(LABELS[idx], float(p)) for p, idx in zip(top_p.tolist(), top_i.tolist())]
    return results, dt

def get_weather_emoji(weather_type):
    emoji_map = {
        'cloudy': '☁️',
        'foggy': '🌫️', 
        'rainy': '🌧️',
        'snowy': '❄️',
        'sunny': '☀️'
    }
    return emoji_map.get(weather_type, '🌤️')

if not files:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px; border: 2px dashed #dee2e6;">
        <h3 style="color: #6c757d;">⬆️ ลากรูปมาวางเพื่อทำนาย</h3>
        <p style="color: #6c757d;">รองรับไฟล์ JPG, PNG และสามารถเลือกหลายไฟล์ได้</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display settings
    col_count = st.slider("จำนวนคอลัมน์ในการแสดงผล", 1, 5, 3)
    cols = st.columns(col_count)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, f in enumerate(files):
        status_text.text(f'กำลังประมวลผลรูปภาพ {i+1}/{len(files)}: {f.name}')
        
        # Load and predict
        img = Image.open(f).convert("RGB")
        res, elapsed = predict_one(img)
        
        # Display results
        with cols[i % len(cols)]:
            # Image display
            st.image(img, caption=f.name, use_column_width=True)
            
            # Prediction results
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 0.5rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: #333;">🎯 ผลการทำนาย</h4>
            """, unsafe_allow_html=True)
            
            for r, (name, p) in enumerate(res, start=1):
                confidence_percent = p * 100
                emoji = get_weather_emoji(name)
                
                # Color coding based on confidence
                if confidence_percent >= 80:
                    color = "#28a745"  # Green
                elif confidence_percent >= 60:
                    color = "#ffc107"  # Yellow
                else:
                    color = "#dc3545"  # Red
                
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                        <span style="font-weight: bold;">{r}. {emoji} {name.title()}</span>
                        <span style="font-weight: bold; color: {color};">{confidence_percent:.1f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_percent}%; background: {color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Timing info
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem; padding: 0.5rem; background: #e9ecef; border-radius: 5px;">
                <small>⏱️ เวลาประมวลผล: {elapsed:.3f} วินาที</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Update progress
        progress_bar.progress((i + 1) / len(files))
    
    # Clear status
    status_text.empty()
    progress_bar.empty()
    
    # Summary
    st.markdown("""
    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 1rem; border-radius: 8px; margin-top: 2rem; text-align: center;">
        <h3 style="margin: 0;">✅ การทำนายเสร็จสิ้น</h3>
        <p style="margin: 0.5rem 0 0 0;">ประมวลผลรูปภาพทั้งหมด {len(files)} รูปเรียบร้อยแล้ว</p>
    </div>
    """, unsafe_allow_html=True)

# Footer Section
st.markdown("---")
st.markdown("""
<div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; margin-top: 3rem;">
    <div style="text-align: center;">
        <h3 style="color: #333; margin-bottom: 1rem;">🤖 เกี่ยวกับ Weather Classifier AI</h3>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 2rem;">
            <div style="flex: 1; min-width: 200px;">
                <h4 style="color: #667eea;">📚 เทคโนโลยี</h4>
                <p style="color: #666; font-size: 0.9rem;">ใช้ PyTorch และ Streamlit ในการพัฒนา<br>รองรับทั้ง CPU และ GPU</p>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <h4 style="color: #667eea;">🎯 ความแม่นยำ</h4>
                <p style="color: #666; font-size: 0.9rem;">ความแม่นยำสูงสุด 96.1%<br>ด้วย Vision Transformer</p>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <h4 style="color: #667eea;">⚡ ประสิทธิภาพ</h4>
                <p style="color: #666; font-size: 0.9rem;">ประมวลผลเร็ว<br>รองรับหลายไฟล์พร้อมกัน</p>
            </div>
        </div>
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #dee2e6;">
            <p style="color: #999; font-size: 0.8rem; margin: 0;">
                © 2024 Weather Classifier AI | พัฒนาด้วย ❤️ โดย AI Technology
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)