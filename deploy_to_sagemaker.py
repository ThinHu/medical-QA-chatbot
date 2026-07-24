import os
import shutil
import sagemaker
import boto3
from sagemaker.pytorch.model import PyTorchModel
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=== BẮT ĐẦU QUÁ TRÌNH DEPLOY VIMQ LÊN SAGEMAKER ===")
    
    # 1. Khởi tạo session AWS
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1")
    role = os.environ.get("SAGEMAKER_EXECUTION_ROLE")
    endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "vimq-ner-intent-endpoint")
    
    if not role:
        print("LỖI: Chưa khai báo SAGEMAKER_EXECUTION_ROLE trong file .env")
        return
        
    print(f"Region: {region}")
    print(f"Role: {role}")
    print(f"Endpoint Name: {endpoint_name}")
    
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # 2. Đóng gói mô hình
    model_dir = "ViMQ_Model"
    deploy_dir = "sagemaker_deploy"
    code_dir = os.path.join(deploy_dir, "code")
    
    print("\n[1/4] Chuẩn bị thư mục deploy...")
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(code_dir, exist_ok=True)
    
    print("Copying model weights...")
    shutil.copy(os.path.join(model_dir, "model.pt"), deploy_dir)
    shutil.copy(os.path.join(model_dir, "training_args.bin"), deploy_dir)
    
    print("Copying inference logic and data...")
    shutil.copytree("src", os.path.join(code_dir, "src"), dirs_exist_ok=True)
    shutil.copytree("data", os.path.join(code_dir, "data"), dirs_exist_ok=True)
    shutil.copy("deploy/inference.py", code_dir)
    shutil.copy("deploy/requirements.txt", code_dir)
    
    # 3. Tạo Sagemaker Model
    print("\n[2/4] Đóng gói và upload lên S3 (có thể mất vài phút)...")
    pytorch_model = PyTorchModel(
        model_data=None, # Tự động pack thư mục deploy_dir
        source_dir=deploy_dir,
        entry_point="code/inference.py",
        role=role,
        framework_version="2.1.0",
        py_version="py310",
        sagemaker_session=sagemaker_session
    )
    
    # 4. Deploy lên Endpoint
    print("\n[3/4] Bắt đầu deploy Endpoint...")
    print(f"CẢNH BÁO: Quá trình này sẽ mất từ 5-10 phút. Đừng tắt cửa sổ này!")
    
    try:
        predictor = pytorch_model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.xlarge", # Máy ảo vừa đủ cho ViMQ (không cần GPU cho inference cơ bản)
            endpoint_name=endpoint_name
        )
        print("\n[4/4] THÀNH CÔNG! Endpoint đã sẵn sàng.")
        print(f"Endpoint Name: {predictor.endpoint_name}")
    except Exception as e:
        print("\n[LỖI] Deploy thất bại:", e)

if __name__ == "__main__":
    main()
