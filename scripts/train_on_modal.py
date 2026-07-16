import os
import modal

app = modal.App("vimq-training-5-iter")

# Định nghĩa container image với các thư viện cần thiết
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "seqeval",
        "scikit-learn",
        "numpy",
        "tqdm",
        "tensorboard"
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("data", remote_path="/root/data")
)

@app.function(
    gpu="T4",
    timeout=7200, # Giới hạn tối đa 2 tiếng
    image=image
)
def run_training_remotely():
    import subprocess
    import shutil
    import sys
    
    print("[Modal Container] Bắt đầu quá trình huấn luyện ViMQ trên GPU...")
    
    model_output_dir = "/root/ViMQ_Model"
    
    # Khởi chạy main.py
    # Hiện tại tôi đang thiết lập num_train_epochs=3 và num_iteration=1 để chạy nghiệm thu vừa đủ.
    # Bạn có thể tăng tham số này lên (như bản gốc là iteration=5) nếu muốn hội tụ sâu hơn.
    command = [
        sys.executable, "src/main.py",
        "--model_type", "vimq_model",
        "--model_dir", model_output_dir,
        "--data_dir", "/root/data",
        "--seed", "100",
        "--do_train",
        "--do_eval",
        "--train_batch_size", "16", # T4 GPU (16GB VRAM) có thể đầy nếu batch=64, nên hạ xuống 16
        "--save_steps", "50",
        "--logging_steps", "50",
        "--num_train_epochs", "3",   
        "--num_iteration", "2",      
        "--tuning_metric", "f1_score",
        "--gpu_id", "0",
        "--iternoise", "1",
        "--omega", "0",
        "--threshold_iou", "0.9",
        "--lamda", "3"
    ]
    
    print(f"[Modal Container] Đang chạy lệnh: {' '.join(command)}")
    result = subprocess.run(command, cwd="/root")
    
    if result.returncode != 0:
        raise Exception("Quá trình huấn luyện gặp lỗi! Vui lòng kiểm tra log bên trên.")
        
    print("[Modal Container] Huấn luyện hoàn tất! Đang nén thư mục mô hình...")
    
    # Nén toàn bộ trọng số thành 1 file zip
    zip_path = "/root/ViMQ_Model_zip"
    shutil.make_archive(zip_path, 'zip', model_output_dir)
    
    # Trả mảng byte về cho local
    with open(f"{zip_path}.zip", "rb") as f:
        model_bytes = f.read()
        
    print("[Modal Container] Đã đóng gói xong mô hình, đang gửi về máy tính của bạn...")
    return model_bytes

@app.local_entrypoint()
def main():
    import zipfile
    import io
    import os
    
    print("[Local] Đang kết nối tới Modal và yêu cầu GPU...")
    model_bytes = run_training_remotely.remote()
    
    print("[Local] Đã nhận được dữ liệu mô hình từ Modal! Đang giải nén...")
    
    local_model_dir = "ViMQ_Model"
    os.makedirs(local_model_dir, exist_ok=True)
    
    with zipfile.ZipFile(io.BytesIO(model_bytes)) as z:
        z.extractall(local_model_dir)
        
    print(f"🚀 [Thành công] Đã lưu trọng số mô hình đã huấn luyện vào thư mục '{local_model_dir}' trong dự án của bạn!")
