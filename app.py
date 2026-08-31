import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io

# ── Page config ──
st.set_page_config(
    page_title="Flood Extent Segmentation",
    page_icon="🌊",
    layout="wide"
)

# ── Class definitions ──
CLASS_NAMES = {
    0: "Background",
    1: "Building Flooded",
    2: "Building Non-Flooded",
    3: "Road Flooded",
    4: "Road Non-Flooded",
    5: "Water",
    6: "Tree",
    7: "Vehicle",
    8: "Pool",
    9: "Grass"
}

CLASS_COLORS = {
    0: "#2C3E50", 1: "#E74C3C", 2: "#E67E22",
    3: "#9B59B6", 4: "#F39C12", 5: "#3498DB",
    6: "#27AE60", 7: "#1ABC9C", 8: "#00CED1",
    9: "#F1C40F"
}

# ── Load model ──
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None
    )
    model.load_state_dict(
        torch.load("flood_model_best.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model

# ── Preprocessing ──
def preprocess(image):
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    img_array = np.array(image.convert("RGB"))
    transformed = transform(image=img_array)
    return transformed['image'].unsqueeze(0)

# ── Prediction ──
def predict(model, image):
    with torch.no_grad():
        output = model(image)
        pred_mask = output.argmax(dim=1).squeeze(0).numpy()
    return pred_mask

# ── Visualization ──
def visualize(original_img, pred_mask):
    cmap = plt.get_cmap('tab10')
    present_classes = np.unique(pred_mask)

    legend_patches = [
        mpatches.Patch(
            color=cmap(cls / 9),
            label=f"{cls}: {CLASS_NAMES[cls]}"
        )
        for cls in present_classes if cls in CLASS_NAMES
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(original_img.resize((512, 512)))
    axes[0].set_title("Original UAV Image", fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(pred_mask, cmap='tab10', vmin=0, vmax=9)
    axes[1].set_title("Predicted Flood Map", fontsize=13)
    axes[1].axis('off')

    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=5,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.05)
    )
    plt.tight_layout()
    return fig

# ── UI ──
st.title("🌊 Flood Extent Segmentation")
st.markdown("**Upload a UAV aerial image to detect flooded areas using deep learning.**")
st.markdown("---")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**Model:** U-Net with ResNet34 encoder

**Dataset:** FloodNet (Hurricane Harvey 2017)

**Classes:** 10 semantic categories

**Training:** 30 epochs on 398 labeled images

**Overall IoU:** 59.9%


""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Class Legend:**")
for cls, name in CLASS_NAMES.items():
    st.sidebar.markdown(f"**{cls}** — {name}")

# Main upload
uploaded_file = st.file_uploader(
    "Upload a UAV aerial image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown("### Processing...")
    with st.spinner("Running flood segmentation model..."):
        model = load_model()
        img_tensor = preprocess(image)
        pred_mask = predict(model, img_tensor)
        fig = visualize(image, pred_mask)

    st.markdown("### Results")
    st.pyplot(fig)

    # Stats
    st.markdown("### Detected Classes")
    present = np.unique(pred_mask)
    cols = st.columns(len(present))
    for i, cls in enumerate(present):
        if cls in CLASS_NAMES:
            pixel_count = (pred_mask == cls).sum()
            percentage = pixel_count / (512 * 512) * 100
            cols[i].metric(
                label=CLASS_NAMES[cls],
                value=f"{percentage:.1f}%"
            )

    st.markdown("---")
    st.success("✅ Segmentation complete!")

else:
    st.info("👆 Upload a UAV aerial image above to get started.")
    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**1️⃣ Upload**\nUpload any UAV aerial image taken after a flood")
    col2.markdown("**2️⃣ Analyze**\nU-Net model analyzes every pixel in the image")
    col3.markdown("**3️⃣ Results**\nSee color-coded flood map with class breakdown")
