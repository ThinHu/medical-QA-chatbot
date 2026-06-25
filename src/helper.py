from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from pinecone_text.sparse import BM25Encoder
from typing import List

# Extract Data From the PDF File
# def load_pdf_file(data):
#     loader= DirectoryLoader(data,
#                             glob="*.pdf",
#                             loader_cls=PyPDFLoader)
#     documents=loader.load()
#     return documents
def load_csv_files(data_dir):
    """
    Load tất cả các file .csv trong thư mục chỉ định.
    Mỗi dòng (row) trong CSV sẽ được load thành 1 Document.
    """
    print(f"[*] Đang quét thư mục '{data_dir}' để tìm các file .csv...")
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.csv",      # Quét tất cả file .csv (kể cả trong thư mục con)
        loader_cls=CSVLoader,
        # Nếu CSV của bạn có format đặc biệt (vd: delimiter là ';', encoding utf-8), 
        # bạn có thể mở comment dòng dưới đây:
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Giữ lại nội dung và metadata 'source' để tránh lỗi khi lưu lên VectorDB.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
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

def get_bm25_encoder():
    """
    Khởi tạo BM25 Encoder để sinh Sparse Vector cho Hybrid Search.
    BM25 rất nhạy với keyword (từ khóa chính xác), bổ khuyết cho Dense Vector.
    """
    # Khởi tạo encoder với thông số mặc định
    bm25_encoder = BM25Encoder().default()
    
    # Lưu ý ở Production: Bạn nên gọi bm25_encoder.fit(corpus) với toàn bộ nội dung PDF 
    # để thuật toán học được phân phối từ vựng, rồi lưu file model ra dạng .json.
    # Trong phạm vi setup nhanh, chúng ta dùng trọng số mặc định.
    return bm25_encoder