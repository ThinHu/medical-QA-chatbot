import os
import time # THÊM IMPORT NÀY ĐỂ DÙNG HÀM SLEEP
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.helper import get_openai_embeddings, load_json_data, filter_to_minimal_docs
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

embeddings = get_openai_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot-advanced" 

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

print("[*] Bắt đầu luồng Data Ingestion...")

# 1. Load JSON Data
print("[*] Đang đọc file JSON từ thư mục data/...")
extracted_data = load_json_data(data_dir='data/')

# 2. Lọc Metadata rác
text_chunks = filter_to_minimal_docs(extracted_data)
print(f"[+] Sử dụng trực tiếp {len(text_chunks)} chunks từ JSON.")

# 3. Kết nối với Vector Store đã tạo
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# 4. Ingest theo Batch (Tránh lỗi Max Tokens và Rate Limit)
batch_size = 100
total_batches = (len(text_chunks) + batch_size - 1) // batch_size

print(f"[*] Bắt đầu đẩy dữ liệu lên Pinecone. Tổng số lô (batches): {total_batches}")

for i in range(0, len(text_chunks), batch_size):
    batch_docs = text_chunks[i : i + batch_size]
    batch_num = (i // batch_size) + 1
    
    print(f"    -> Đang đẩy Lô {batch_num}/{total_batches} (từ dòng {i} đến {i + len(batch_docs)})...")
    
    docsearch.add_documents(documents=batch_docs)
    
    time.sleep(2)

print("[SUCCESS] Đã đẩy toàn bộ dữ liệu JSON lên Pinecone thành công!")