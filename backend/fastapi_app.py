# fastapi_app.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import grpc
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.grpc_protos.chat_pb2 as chat_pb2
import backend.grpc_protos.chat_pb2_grpc as chat_pb2_grpc

app = FastAPI()

async def grpc_stream_generator(session_id: str, message: str):
    """Mở luồng gRPC và yield dữ liệu chuẩn SSE, có bắt lỗi"""
    try:
        async with grpc.aio.insecure_channel('localhost:50051') as channel:
            stub = chat_pb2_grpc.LangGraphServiceStub(channel)
            request = chat_pb2.ChatRequest(session_id=session_id, message=message)
            
            import json
            async for chunk in stub.StreamChat(request):
                encoded_token = json.dumps(chunk.token)
                yield f"data: {encoded_token}\n\n"
    
    except Exception as e:
        # IN LỖI RA TERMINAL 2 ĐỂ BẠN BIẾT ĐƯỜNG DEBUG
        print(f"LỖI KẾT NỐI gRPC: {e}") 
        # Bắn lỗi lên màn hình Chainlit cho user thấy
        yield f"data: [Lỗi hệ thống Backend: Không thể xử lý yêu cầu]\n\n"
            
@app.get("/api/chat")
async def chat_endpoint(session_id: str, message: str):
    # Trả về StreamingResponse thay vì JSON thông thường
    return StreamingResponse(
        grpc_stream_generator(session_id, message), 
        media_type="text/event-stream"
    )

# Chạy server bằng lệnh: uvicorn fastapi_app:app --port 8000