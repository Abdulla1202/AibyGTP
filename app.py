from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import json
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

from agent import get_agent
from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations,
    list_all_conversations,
    list_conversations_by_user,
    get_conversation_owner,
    get_admin_stats,
    count_user_conversations,
    delete_conversation,
    count_messages_in_thread,
    migrate_guest_chat
)

from rag import add_document_to_rag
from tools import set_current_thread_id
from auth import (
    User,
    register_user,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_optional_current_user,
    require_admin,
    create_default_admin,
    get_all_users
)


app = FastAPI()

templates = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


init_db()
create_default_admin()


# ═══════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/auth/register")
async def auth_register(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON."}, status_code=400)

    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not username or not email or not password:
        return JSONResponse(
            {"error": "Username, email, and password are required."},
            status_code=400
        )

    if len(username) < 3:
        return JSONResponse(
            {"error": "Username must be at least 3 characters."},
            status_code=400
        )

    if len(password) < 4:
        return JSONResponse(
            {"error": "Password must be at least 4 characters."},
            status_code=400
        )

    try:
        user = register_user(username, email, password)
        user_id = user.id

        # Migrate guest chat if provided
        guest_thread_id = data.get("guest_thread_id")
        if guest_thread_id:
            migrate_guest_chat(guest_thread_id, user_id)

        token = create_access_token({
            "user_id": user_id,
            "username": user.username,
            "is_admin": user.is_admin
        })

        return {
            "token": token,
            "user": {
                "user_id": user.id,
                "username": user.username,
                "is_admin": user.is_admin
            }
        }

    except Exception as e:
        detail = getattr(e, "detail", str(e))
        return JSONResponse({"error": detail}, status_code=400)


@app.post("/auth/login")
async def auth_login(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON."}, status_code=400)

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return JSONResponse(
            {"error": "Username and password are required."},
            status_code=400
        )

    user = authenticate_user(username, password)

    if not user:
        return JSONResponse(
            {"error": "Invalid username or password."},
            status_code=401
        )

    # Migrate guest chat if provided
    guest_thread_id = data.get("guest_thread_id")
    if guest_thread_id:
        migrate_guest_chat(guest_thread_id, user.id)

    token = create_access_token({
        "user_id": user.id,
        "username": user.username,
        "is_admin": user.is_admin
    })

    return {
        "token": token,
        "user": {
            "user_id": user.id,
            "username": user.username,
            "is_admin": user.is_admin
        }
    }


@app.get("/auth/me")
async def auth_me(current_user: dict = Depends(get_current_user)):
    return current_user


# ═══════════════════════════════════════════════════════════════
# PAGE ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={}
    )


@app.get("/admin")
async def admin_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={}
    )


# ═══════════════════════════════════════════════════════════════
# CHAT ENDPOINTS (Protected)
# ═══════════════════════════════════════════════════════════════

@app.get("/conversations")
async def conversations(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    items = list_conversations(user_id=user_id)

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat() if item.created_at else "",
                "updated_at": item.updated_at.isoformat() if item.updated_at else ""
            }
            for item in items
        ]
    }



@app.get("/history/{thread_id}")
async def history(thread_id: str, current_user: dict = Depends(get_current_user)):
    # Check ownership: user can only view their own conversations
    owner_id = get_conversation_owner(thread_id)
    if owner_id is not None and owner_id != current_user["user_id"]:
        if not current_user.get("is_admin"):
            return JSONResponse(
                {"error": "You don't have access to this conversation."},
                status_code=403
            )

    messages = get_chat_history(thread_id)

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }



@app.delete("/conversations/{thread_id}")
async def delete_conv(thread_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    is_admin = current_user.get("is_admin", False)
    result = delete_conversation(thread_id, user_id=user_id, is_admin=is_admin)
    if result:
        return {"success": True, "message": "Conversation deleted."}
    return JSONResponse(
        {"success": False, "message": "Not found or no permission."},
        status_code=403
    )


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user["user_id"]
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]

        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV."
                },
                status_code=400
            )

        file_id = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        create_or_update_conversation(thread_id, "Uploaded document", user_id=user_id)

        # Process document in background so user doesn't have to wait
        background_tasks.add_task(add_document_to_rag, file_path, thread_id)

        return JSONResponse({
            "success": True,
            "message": f"Uploaded {filename} successfully! We are now processing it in the background. You can start chatting soon."
        })

    except Exception as e:
        import logging
        logging.error(f"Upload Error: {str(e)}", exc_info=True)
        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )



def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def should_stream_chunk(chunk, metadata) -> bool:
    """
    This prevents raw tool/search/RAG JSON from appearing in the frontend.

    We only stream normal AI text chunks.
    We do NOT stream:
    - ToolMessage
    - messages from tool nodes
    - tool call chunks
    - raw tool outputs
    """

    metadata = metadata or {}

    node_name = str(metadata.get("langgraph_node", "")).lower()

    if "tool" in node_name:
        return False

    if isinstance(chunk, ToolMessage):
        return False

    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}

    if additional_kwargs.get("tool_calls"):
        return False

    return True


def extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        return "".join(text_parts)

    return ""



@app.post("/chat/stream")
async def chat_stream(request: Request, current_user: dict = Depends(get_optional_current_user)):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body."},
            status_code=400
        )

    user_id = current_user["user_id"] if current_user else None
    user_message = data.get("message", "")
    thread_id = data.get("thread_id", "default")
    selected_model = data.get("model", "gemini-2.5-flash")

    if not user_message.strip():
        return JSONResponse(
            {"error": "Message is required."},
            status_code=400
        )

    # Guest Limit Check
    if user_id is None:
        # If guest, count messages for this thread
        msg_count = count_messages_in_thread(thread_id)
        if msg_count >= 5:
            return JSONResponse(
                {"error": "limit_reached", "message": "Aapki free limit khatam ho gayi hai. Aage baat karne ke liye please login karein."},
                status_code=403
            )

    agent = get_agent(selected_model)

    create_or_update_conversation(thread_id, user_message, user_id=user_id)
    save_chat_message(thread_id, "user", user_message, user_id=user_id)

    set_current_thread_id(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    def event_generator():
        final_answer = ""

        try:
            inputs = {
                "messages": [
                    HumanMessage(content=user_message)
                ]
            }

            yield sse_data({"thinking": f"Sending to {selected_model}..."})

            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages"
            ):
                metadata = metadata or {}
                node_name = str(metadata.get("langgraph_node", "")).lower()

                # Stream tool call events (when agent decides to use a tool)
                if isinstance(chunk, (AIMessage, AIMessageChunk)):
                    tool_calls = getattr(chunk, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            tool_name = tc.get("name", "unknown")
                            tool_name_display = tool_name.replace("_", " ").title()
                            yield sse_data({"tool_call": tool_name_display})
                        continue

                # Stream tool result events (when tool returns result)
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    tool_name_display = tool_name.replace("_", " ").title()
                    yield sse_data({"tool_result": tool_name_display})
                    continue

                # Skip non-AI chunks from tool nodes
                if "tool" in node_name:
                    continue

                if not isinstance(chunk, (AIMessage, AIMessageChunk)):
                    continue

                if getattr(chunk, "invalid_tool_calls", None):
                    continue

                additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
                if additional_kwargs.get("tool_calls"):
                    continue

                token = extract_text_from_chunk(chunk)

                if token:
                    final_answer += token
                    yield sse_data({"token": token})

            # Check for HITL interrupt after streaming
            try:
                state = agent.get_state(config)
                if hasattr(state, 'tasks') and state.tasks:
                    for task in state.tasks:
                        if hasattr(task, 'interrupts') and task.interrupts:
                            for intr in task.interrupts:
                                interrupt_value = getattr(intr, 'value', str(intr))
                                yield sse_data({
                                    "hitl": {
                                        "message": str(interrupt_value),
                                        "thread_id": thread_id
                                    }
                                })
            except Exception:
                pass

            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer, user_id=user_id)

            yield sse_data({"done": True})

        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str or "too many" in error_str:
                friendly = "Server busy hai, thoda wait karein aur dobara try karein."
            elif "model" in error_str and ("not found" in error_str or "not exist" in error_str):
                friendly = "Selected model available nahi hai. Doosra model select karein."
            elif "401" in error_str or "authentication" in error_str or "api_key" in error_str:
                friendly = "API key issue hai. Admin se contact karein."
            elif "timeout" in error_str or "timed out" in error_str:
                friendly = "Response mein time lag raha hai. Dobara try karein."
            else:
                friendly = "Kuch gadbad ho gayi. Please dobara try karein."
            yield sse_data({"error": friendly})
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ═══════════════════════════════════════════════════════════════
# HITL CONFIRM ENDPOINT
# ═══════════════════════════════════════════════════════════════

@app.post("/chat/confirm")
async def chat_confirm(request: Request, current_user: dict = Depends(get_current_user)):
    from langgraph.types import Command

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    thread_id = data.get("thread_id", "")
    decision = data.get("decision", "no")
    selected_model = data.get("model", "qwen/qwen3.8-27b")
    user_id = current_user["user_id"]

    if not thread_id:
        return JSONResponse({"error": "thread_id required"}, status_code=400)

    agent = get_agent(selected_model)
    config = {"configurable": {"thread_id": thread_id}}
    set_current_thread_id(thread_id)

    def event_generator():
        final_answer = ""
        try:
            yield sse_data({"thinking": f"Processing your {'approval' if decision == 'yes' else 'rejection'}..."})

            for chunk, metadata in agent.stream(
                Command(resume=decision),
                config=config,
                stream_mode="messages"
            ):
                metadata = metadata or {}
                node_name = str(metadata.get("langgraph_node", "")).lower()

                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    yield sse_data({"tool_result": tool_name.replace("_", " ").title()})
                    continue

                if "tool" in node_name:
                    continue

                if not isinstance(chunk, (AIMessage, AIMessageChunk)):
                    continue

                if getattr(chunk, "tool_calls", None):
                    continue

                additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
                if additional_kwargs.get("tool_calls"):
                    continue

                token = extract_text_from_chunk(chunk)
                if token:
                    final_answer += token
                    yield sse_data({"token": token})

            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer, user_id=user_id)

            yield sse_data({"done": True})

        except Exception as e:
            yield sse_data({"error": "HITL processing failed. Try again."})
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# ═══════════════════════════════════════════════════════════════
# ADMIN ENDPOINTS (Protected + Admin Only)
# ═══════════════════════════════════════════════════════════════

@app.get("/admin/stats")
async def admin_stats(current_user: dict = Depends(require_admin)):
    stats = get_admin_stats()
    return stats


@app.get("/admin/users")
async def admin_users(current_user: dict = Depends(require_admin)):
    users = get_all_users()

    return {
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "is_admin": u.is_admin,
                "created_at": u.created_at.isoformat() if u.created_at else "",
                "conversation_count": count_user_conversations(u.id)
            }
            for u in users
        ]
    }


@app.get("/admin/conversations")
async def admin_all_conversations(current_user: dict = Depends(require_admin)):
    items = list_all_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "user_id": item.user_id,
                "title": item.title,
                "created_at": item.created_at.isoformat() if item.created_at else "",
                "updated_at": item.updated_at.isoformat() if item.updated_at else ""
            }
            for item in items
        ]
    }


@app.get("/admin/conversations/{user_id}")
async def admin_user_conversations(user_id: int, current_user: dict = Depends(require_admin)):
    items = list_conversations_by_user(user_id)

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat() if item.created_at else "",
                "updated_at": item.updated_at.isoformat() if item.updated_at else ""
            }
            for item in items
        ]
    }


@app.get("/admin/history/{thread_id}")
async def admin_history(thread_id: str, current_user: dict = Depends(require_admin)):
    messages = get_chat_history(thread_id)

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else ""
            }
            for msg in messages
        ]
    }


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
   
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8090)),
        reload=False
    )