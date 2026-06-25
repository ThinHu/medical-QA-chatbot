import chainlit as cl
import httpx

FASTAPI_URL = "http://localhost:8080/api/chat"

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("id")
    
    # Khởi tạo một tin nhắn rỗng trên màn hình
    msg = cl.Message(content="")
    await msg.send()

    # Mở HTTP Client để hứng stream từ FastAPI
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "GET", 
            FASTAPI_URL,
            params={"session_id": session_id, "message": message.content}
        ) as response:
            import json
            # Lặp qua từng dòng dữ liệu SSE trả về
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    # Lấy token thật (cắt bỏ chữ "data: ")
                    token_str = line.replace("data: ", "", 1)
                    
                    try:
                        token = json.loads(token_str)
                    except json.JSONDecodeError:
                        token = token_str
                    
                    # Bắn token lên màn hình Chainlit
                    await msg.stream_token(token)

    # Cập nhật trạng thái hoàn thành tin nhắn
    await msg.update()