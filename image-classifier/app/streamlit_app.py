import streamlit as st
import torch
import yaml
from PIL import Image
from torchvision import transforms
import os
import io
from src.model import build_model

@st.cache_resource
def load_model_and_labels(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    ckpt_path = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["classes"]
    model = build_model(cfg["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_names, cfg

def main():
    st.title("🌿 Medicinal Plant Image Classifier")
    st.write("Upload a plant image to identify its species.")

    model, class_names, cfg = load_model_and_labels("config.yaml")
    tfm = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor()
    ])

    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if file is not None:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)
        x = tfm(image).unsqueeze(0)
        with torch.no_grad():
            probs = model(x).softmax(dim=1).squeeze().tolist()

        # Show top-3
        top3 = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)[:3]
        st.subheader("Prediction")
        for idx, p in top3:
            st.write(f"- **{class_names[idx]}** — {p*100:.2f}%")

if __name__ == "__main__":
    main()
