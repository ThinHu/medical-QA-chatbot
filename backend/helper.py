from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from typing import List

# Extract Data From the PDF File
# def load_pdf_file(data):
#     loader= DirectoryLoader(data,
#                             glob="*.pdf",
#                             loader_cls=PyPDFLoader)
#     documents=loader.load()
#     return documents
import json
from pathlib import Path

def load_json_data(data_dir):
    """
    Load ViMQ JSON files and convert to LangChain Documents.
    """
    print(f"[*] Đang quét thư mục '{data_dir}' để tìm các file .json...")
    documents = []
    data_path = Path(data_dir)
    for json_file in data_path.glob("*.json"):
        if json_file.name in ["char2index.json"]:
            continue
        print(f"    -> Đọc file: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "sentence" in item:
                            doc = Document(
                                page_content=item["sentence"],
                                metadata={
                                    "source": f"ViMQ/{json_file.name}",
                                    "intent": item.get("sent_label", "")
                                }
                            )
                            documents.append(doc)
            except Exception as e:
                print(f"Lỗi khi đọc file {json_file.name}: {e}")
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Giữ lại nội dung và metadata 'source', 'intent'.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        intent = doc.metadata.get("intent", "")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src, "intent": intent}
            )
        )
    return minimal_docs



# #Split the Data into Text Chunks
# def text_split(extracted_data):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     text_chunks=text_splitter.split_documents(extracted_data)
#     return text_chunks

def get_semantic_chunks(documents, embeddings):
    """
    Chia nhỏ văn bản dựa trên sự thay đổi ngữ nghĩa (Semantic Chunking).
    Nó sẽ so sánh cosine similarity giữa các câu liên tiếp, nếu độ lệch lớn hơn ngưỡng, 
    nó sẽ ngắt thành một chunk mới.
    """
    print("[*] Khởi tạo Semantic Chunker...")
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile" # Cắt khi sự thay đổi ngữ nghĩa vượt quá phân vị mặc định (95%)
    )
    
    print("[*] Đang tính toán và cắt nội dung (có thể tốn thời gian do gọi API Embedding)...")
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


#Download the Embeddings from HuggingFace 
# def download_hugging_face_embeddings():
#     embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
#     return embeddings

def get_openai_embeddings():
    """
    Sử dụng OpenAI text-embedding-3-small.
    Đảm bảo biến môi trường OPENAI_API_KEY đã được load.
    Model này trả về vector 1536 dimensions (mặc định), tối ưu chi phí và hiệu suất hơn MiniLM.
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")
