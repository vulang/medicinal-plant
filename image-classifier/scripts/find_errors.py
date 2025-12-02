# scripts/find_errors.py
import sys
import os
import torch
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import build_testloader
from src.model import build_model
from src.utils import resolve_device

@torch.no_grad()
def get_predictions_with_paths(model, loader, device):
    model.eval()
    results = []
    
    if hasattr(loader.dataset, 'samples'):
        img_paths = [s[0] for s in loader.dataset.samples]
    else:
        raise ValueError("Dataset không hỗ trợ lấy đường dẫn file.")
        
    idx = 0
    for images, labels in tqdm(loader, desc="Scanning Errors", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        
        confidences, preds = torch.max(probs, dim=1)
        
        preds = preds.cpu().tolist()
        labels = labels.cpu().tolist()
        confidences = confidences.cpu().tolist()
        
        for i in range(len(preds)):
            if preds[i] != labels[i]: # Chỉ quan tâm ảnh đoán sai
                results.append({
                    "file_path": img_paths[idx],
                    "true_label_idx": labels[i],
                    "pred_label_idx": preds[i],
                    "confidence": confidences[i] # Độ tự tin của mô hình vào cái sai
                })
            idx += 1
            
    return results

def main():
    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    device = resolve_device(cfg.get("device", "auto"))
    
    # Load model tốt nhất
    best_ckpt = os.path.join(cfg["save_dir"], f"{cfg['model_name']}_best.pt")
    ckpt = torch.load(best_ckpt, map_location=device)
    class_names = ckpt["classes"]
    
    model = build_model(cfg["model_name"], len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    
    # Load Test Loader (Dùng test set hoặc val set để kiểm tra)
    # Mẹo: Bạn có thể đổi cfg['test_dir'] thành cfg['train_dir'] tạm thời 
    # trong config.yaml để quét lỗi trên tập TRAIN (nơi quan trọng nhất)
    data_cfg = getattr(model, "data_config", None)
    loader = build_testloader(
        cfg["train_dir"], # <--- QUÉT TRÊN TẬP TRAIN
        cfg["img_size"],
        cfg["batch_size"],
        cfg["num_workers"],
        class_names,
        data_cfg=data_cfg
    )
    
    print("Đang quét dữ liệu tìm ảnh lỗi...")
    errors = get_predictions_with_paths(model, loader, device)
    
    # Chuyển thành DataFrame để dễ xem
    df = pd.DataFrame(errors)
    
    # Thêm tên lớp cho dễ đọc
    df["true_label_name"] = df["true_label_idx"].apply(lambda x: class_names[x])
    df["pred_label_name"] = df["pred_label_idx"].apply(lambda x: class_names[x])
    
    # Sắp xếp: Những ảnh mô hình RẤT TỰ TIN là sai -> Khả năng cao là nhãn sai
    df = df.sort_values(by="confidence", ascending=False)
    
    # Lưu ra file CSV
    output_file = "potential_label_errors.csv"
    df.to_csv(output_file, index=False)
    print(f"\nĐã tìm thấy {len(df)} ảnh nghi vấn.")
    print(f"Danh sách đã lưu tại: {output_file}")
    print("Hãy mở file CSV, kiểm tra cột 'file_path' của những dòng đầu tiên!")

if __name__ == "__main__":
    main()