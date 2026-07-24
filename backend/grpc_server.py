import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys

# Force UTF-8 output and ensure it doesn't get fully buffered
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import asyncio
import grpc
from dotenv import load_dotenv
load_dotenv()
import psycopg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.grpc_protos.chat_pb2 as chat_pb2
import backend.grpc_protos.chat_pb2_grpc as chat_pb2_grpc

import logging
import watchtower
import boto3

# Setup CloudWatch Logger
logger = logging.getLogger("ViMQ")
logger.setLevel(logging.INFO)

if os.environ.get("AWS_ACCESS_KEY_ID"):
    boto3_session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    cw_handler = watchtower.CloudWatchLogHandler(
        boto3_client=boto3_session.client("logs"),
        log_group_name="med-chatbot",
        log_stream_name="gRPC-Server"
    )
    logger.addHandler(cw_handler)
else:
    logger.addHandler(logging.StreamHandler())

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

from backend.helper import get_openai_embeddings
from backend.prompt import (
    contextualize_q_prompt, 
    router_prompt, 
    treatment_prompt, 
    cause_prompt, 
    severity_prompt, 
    diagnosis_prompt, 
    other_prompt
)

load_dotenv()

# --- FIX LỖI EVENT LOOP TRÊN WINDOWS ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def build_rag_chain():
    """Hàm khởi tạo toàn bộ bộ não AI và kết nối Database"""
    logger.info("[Debug] Kết nối Postgres DB...")
    DB_URI = os.environ.get('DB_URI')
    if "keepalives" not in DB_URI:
        separator = "&" if "?" in DB_URI else "?"
        DB_URI += f"{separator}keepalives=1&keepalives_idle=60&keepalives_interval=10&keepalives_count=5"
        
    async_connection = await psycopg.AsyncConnection.connect(DB_URI, autocommit=True)
    
    class SafePostgresChatMessageHistory(PostgresChatMessageHistory):
        async def _ensure_connection(self):
            if self._aconnection.closed:
                logger.warning("Connection closed. Reconnecting...")
                self._aconnection = await psycopg.AsyncConnection.connect(DB_URI, autocommit=True)

        async def aget_messages(self):
            try:
                await self._ensure_connection()
                return await super().aget_messages()
            except psycopg.OperationalError:
                logger.warning("OperationalError on get. Reconnecting...")
                self._aconnection = await psycopg.AsyncConnection.connect(DB_URI, autocommit=True)
                return await super().aget_messages()
                
        async def aadd_messages(self, messages):
            try:
                await self._ensure_connection()
                return await super().aadd_messages(messages)
            except psycopg.OperationalError:
                logger.warning("OperationalError on add. Reconnecting...")
                self._aconnection = await psycopg.AsyncConnection.connect(DB_URI, autocommit=True)
                return await super().aadd_messages(messages)

    # Tạo bảng nếu chưa có
    logger.info("[Debug] Tạo bảng chat_history...")
    await SafePostgresChatMessageHistory.acreate_tables(async_connection, "chat_history")

    def get_session_history(session_id: str):
        return SafePostgresChatMessageHistory(
            "chat_history", 
            session_id, 
            async_connection=async_connection
        )

    logger.info("[Debug] Khởi tạo Embeddings...")
    embeddings = get_openai_embeddings()
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True) 

    logger.info("[Debug] Khởi tạo Pinecone...")
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    index_name = "medical-chatbot-advanced"
    pinecone_index = pc.Index(index_name)
    logger.info("[Debug] Kết nối Pinecone Index...")
    vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    logger.info("[Debug] Khởi tạo Cohere Rerank...")
    cohere_rerank = CohereRerank(cohere_api_key=os.environ.get('COHERE_API_KEY'), model="rerank-multilingual-v3.0", top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=cohere_rerank, base_retriever=base_retriever)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableBranch

    history_aware_retriever = contextualize_q_prompt | chat_model | StrOutputParser() | compression_retriever
    
    treatment_chain = create_stuff_documents_chain(chat_model, treatment_prompt)
    cause_chain = create_stuff_documents_chain(chat_model, cause_prompt)
    severity_chain = create_stuff_documents_chain(chat_model, severity_prompt)
    diagnosis_chain = create_stuff_documents_chain(chat_model, diagnosis_prompt)
    other_chain = create_stuff_documents_chain(chat_model, other_prompt)

    question_answer_chain = RunnableBranch(
        (lambda x: "TREATMENT" in x.get("vimq_intent", "").upper(), treatment_chain),
        (lambda x: "CAUSE" in x.get("vimq_intent", "").upper(), cause_chain),
        (lambda x: "SEVERITY" in x.get("vimq_intent", "").upper(), severity_chain),
        (lambda x: "DIAGNOSIS" in x.get("vimq_intent", "").upper(), diagnosis_chain),
        other_chain
    )
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain, chat_model


class LangGraphServicer(chat_pb2_grpc.LangGraphServiceServicer):
    def __init__(self, chain, chat_model):
        self.chain = chain
        self.chat_model = chat_model

    async def StreamChat(self, request, context):
        """Nhận request từ FastAPI và stream từng token trả về"""
        session_id = request.session_id
        user_message = request.message
        logger.info(f"Nhận luồng chat mới - Session: {session_id}")

        from backend.vimq_integration import analyze_query
        vimq_result = analyze_query(user_message)
        entities = ", ".join(vimq_result.get("entities", []))
        if not entities:
            entities = "Không có"

        from langchain_core.output_parsers import StrOutputParser
        
        # Dùng LLM để phân loại Intent chuẩn xác hơn hardcode
        intent_chain = router_prompt | self.chat_model | StrOutputParser()
        intent = await intent_chain.ainvoke({"input": user_message, "vimq_entities": entities})
        logger.info(f"[IC Router] Classified Intent: {intent}")

        langfuse_handler = CallbackHandler()

        async for chunk in self.chain.astream(
            {
                "input": user_message,
                "vimq_intent": intent,
                "vimq_entities": entities
            },
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
    logger.info("Đang khởi tạo LangChain & Vector DB...")
    chain, chat_model = await build_rag_chain()
    
    server = grpc.aio.server()
    chat_pb2_grpc.add_LangGraphServiceServicer_to_server(LangGraphServicer(chain, chat_model), server)
    server.add_insecure_port('[::]:50051')
    
    logger.info("gRPC LangGraph Server đang chạy tại port 50051...")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())