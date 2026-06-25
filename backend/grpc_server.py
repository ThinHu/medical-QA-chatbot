import os
import sys
import asyncio
import grpc
from dotenv import load_dotenv
import psycopg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.grpc_protos.chat_pb2 as chat_pb2
import backend.grpc_protos.chat_pb2_grpc as chat_pb2_grpc

# LangChain Core & Models
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres import PostgresChatMessageHistory

# Retrievers
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langfuse.langchain import CallbackHandler

from src.helper import get_openai_embeddings
from src.prompt import contextualize_q_prompt, qa_prompt

load_dotenv()

# --- FIX LỖI EVENT LOOP TRÊN WINDOWS ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def build_rag_chain():
    """Hàm khởi tạo toàn bộ bộ não AI và kết nối Database"""
    DB_URI = os.environ.get('DB_URI')
    async_connection = await psycopg.AsyncConnection.connect(DB_URI)
    
    # Tạo bảng chat_history nếu chưa có
    await PostgresChatMessageHistory.acreate_tables(async_connection, "chat_history")

    def get_session_history(session_id: str):
        return PostgresChatMessageHistory(
            "chat_history", 
            session_id, 
            async_connection=async_connection
        )

    # Khởi tạo các thành phần LangChain
    embeddings = get_openai_embeddings()
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True) 

    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    vectorstore = PineconeVectorStore.from_existing_index("medical-chatbot-advanced", embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    cohere_rerank = CohereRerank(cohere_api_key=os.environ.get('COHERE_API_KEY'), model="rerank-multilingual-v3.0", top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=cohere_rerank, base_retriever=base_retriever)

    from langchain_core.output_parsers import StrOutputParser
    history_aware_retriever = contextualize_q_prompt | chat_model | StrOutputParser() | compression_retriever
    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


class LangGraphServicer(chat_pb2_grpc.LangGraphServiceServicer):
    def __init__(self, chain):
        self.chain = chain

    async def StreamChat(self, request, context):
        """Nhận request từ FastAPI và stream từng token trả về"""
        session_id = request.session_id
        user_message = request.message

        print(f"Nhận luồng chat mới - Session: {session_id}")

        langfuse_handler = CallbackHandler()

        async for chunk in self.chain.astream(
            {"input": user_message},
            config={
                "configurable": {"session_id": session_id},
                "callbacks": [langfuse_handler],
                "metadata": {"session_id": session_id}
            }
        ):
            # Với cấu trúc create_retrieval_chain, kết quả sinh ra nằm ở key 'answer'
            if "answer" in chunk:
                # Bắn token qua gRPC ngay lập tức
                yield chat_pb2.ChatChunk(token=chunk["answer"])

async def serve():
    # Khởi tạo RAG Chain trước khi mở server
    print("Đang khởi tạo LangChain & Vector DB...")
    chain = await build_rag_chain()
    
    server = grpc.aio.server()
    chat_pb2_grpc.add_LangGraphServiceServicer_to_server(LangGraphServicer(chain), server)
    server.add_insecure_port('[::]:50051')
    
    print("gRPC LangGraph Server đang chạy tại port 50051...")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())