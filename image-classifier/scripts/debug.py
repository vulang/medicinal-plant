# scripts/debug_class_25.py
import sys
import os
import torch
import yaml
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Thêm đường dẫn để import được các module trong src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import build_transforms, _is_valid_image
from src.model import build_model
from src.utils import resolve_device

def main(target_class_name="25"):
    # 1. Load cấu hình
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("Lỗi: Không tìm thấy file config.yaml")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg.get("device", "auto"))
    print(f"Đang sử dụng thiết bị: {device}")

    # 2. Load Model tốt nhất (Swin hoặc ConvNeXt tùy config)
    # Lưu ý: Đảm bảo model_name trong config khớp với file checkpoint bạn muốn kiểm tra
    model_name = cfg['model_name']
    best_ckpt_path = os.path.join(cfg["save_dir"], f"{model_name}_best.pt")
    
    if not os.path.exists(best_ckpt_path):
        print(f"Lỗi: Không tìm thấy checkpoint tại {best_ckpt_path}")
        print("Hãy kiểm tra lại model_name trong config.yaml")
        return

    print(f"Đang load model: {model_name}...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    class_names = ckpt["classes"]
    
    # Kiểm tra xem lớp mục tiêu có tồn tại không
    if target_class_name not in class_names:
        print(f"Lỗi: Lớp '{target_class_name}' không tồn tại trong danh sách lớp đã train.")
        return
        
    target_class_idx = class_names.index(target_class_name)
    
    model = build_model(model_name, len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3. Chuẩn bị dữ liệu (Chỉ lấy ảnh của lớp 25)
    test_dir = cfg["test_dir"]
    target_dir = os.path.join(test_dir, target_class_name)
    
    if not os.path.exists(target_dir):
        print(f"Lỗi: Không tìm thấy thư mục ảnh test: {target_dir}")
        return

    # Lấy danh sách file ảnh (kiểm tra bằng path đầy đủ để không báo sai lệch)
    image_files = [
        f for f in os.listdir(target_dir)
        if _is_valid_image(os.path.join(target_dir, f))
    ]
    print(f"Tìm thấy {len(image_files)} ảnh trong thư mục lớp {target_class_name}.")

    # Pipeline xử lý ảnh (giống hệt lúc validation)
    transform = build_transforms(
        img_size=cfg["img_size"], 
        is_train=False, 
        data_cfg=getattr(model, "data_config", None)
    )

    # 4. Chạy dự đoán và tìm lỗi
    errors = []
    
    print("Đang phân tích...")
    with torch.no_grad():
        for img_name in tqdm(image_files):
            img_path = os.path.join(target_dir, img_name)
            
            try:
                # Load và transform ảnh
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device) # Thêm batch dim
                
                # Dự đoán
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                # Lấy Top 1
                conf, pred_idx = torch.max(probs, dim=1)
                pred_idx = pred_idx.item()
                conf = conf.item()
                
                # Nếu đoán sai (Khác lớp 25)
                if pred_idx != target_class_idx:
                    errors.append({
                        "filename": img_name,
                        "path": img_path,
                        "predicted_class": class_names[pred_idx],
                        "confidence": conf  # Độ tự tin vào cái sai
                    })
            except Exception as e:
                print(f"Lỗi khi đọc file {img_name}: {e}")

    # 5. Báo cáo kết quả
    print("\n" + "="*50)
    print(f"KẾT QUẢ PHÂN TÍCH LỚP {target_class_name}")
    print("="*50)
    
    if not errors:
        print("Tuyệt vời! Không có ảnh nào bị đoán sai.")
    else:
        print(f"Tổng số ảnh đoán sai: {len(errors)}/{len(image_files)} "
              f"({len(errors)/len(image_files)*100:.1f}%)")
        
        # Sắp xếp theo độ tự tin giảm dần (Sai mà tự tin nhất xếp đầu)
        errors.sort(key=lambda x: x['confidence'], reverse=True)
        
        print("\nTOP 5 ẢNH BỊ NHẦM LẪN NGHIÊM TRỌNG NHẤT:")
        for i, err in enumerate(errors[:5]):
            print(f"{i+1}. {err['filename']}")
            print(f"   -> Bị nhầm thành: {err['predicted_class']} (Độ tin cậy: {err['confidence']:.2%})")
            print(f"   -> Đường dẫn: {err['path']}")
            print("-" * 30)
            
        # Lưu danh sách đầy đủ ra CSV để bạn xem sau
        df = pd.DataFrame(errors)
        csv_path = f"errors_class_{target_class_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nĐã lưu danh sách lỗi đầy đủ vào file: {csv_path}")

if __name__ == "__main__":
    main(target_class_name="97")
