import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from tools import tools

Path("data").mkdir(exist_ok=True)


# Groq available models (verified)
DEFAULT_MODEL = "qwen/qwen3.8-27b"

ALLOWED_MODELS = {
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "allam-2-7b",
}


SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named AibyAI.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool (search_uploaded_documents).
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.
8. Check current weather for any city using the weather tool.
9. Get real-time stock prices and market data using the stock price tool.
10. Purchase stocks (with human confirmation) using purchase_stock tool.

Rules:
- IMPORTANT: If the user asks about an uploaded document, a PDF, a resume, a file, or says "tell me about this pdf", you MUST call the 'search_uploaded_documents' tool immediately. Do NOT tell the user you can't see the file; instead, use the tool to search for it.
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use Tavily Search.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- If the user asks about weather, temperature, climate, or conditions for any city, use get_weather.
- If the user asks about stock prices, shares, market data, or financial info for any company, use get_stock_price. Use the correct ticker symbol (e.g., AAPL for Apple, GOOGL for Google, MSFT for Microsoft, TSLA for Tesla).
- If the user wants to BUY or PURCHASE stocks, use purchase_stock. This requires human confirmation.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
- Format your responses using Markdown for better readability (use **bold**, headings, lists, code blocks, tables etc where appropriate).
"""


def normalize_model_name(model_name: str | None) -> str:
    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


def build_agent(model_name: str):
    """
    Build one LangGraph agent for a selected Groq model.
    """

    selected_model = normalize_model_name(model_name)

    llm = ChatGroq(
        model=selected_model,
        temperature=0.3,
        streaming=True,
        max_tokens=800
    )

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]
