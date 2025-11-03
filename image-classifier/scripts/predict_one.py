import argparse
import yaml
import torch
from PIL import Image
from torchvision import transforms
import os
from src.model import build_model

def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_best.pt")
    ckpt = load_checkpoint(ckpt_path, device)
    class_names = ckpt["classes"]
    model = build_model(cfg["model_name"], len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([transforms.Resize((cfg["img_size"], cfg["img_size"])), transforms.ToTensor()])
    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(dim=1).cpu().squeeze().tolist()
    top_idx = int(torch.tensor(probs).argmax().item())
    print(f"Predicted: {class_names[top_idx]} (p={probs[top_idx]:.4f})")

if __name__ == "__main__":
    main()
