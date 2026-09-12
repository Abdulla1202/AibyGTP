import os
import math
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from database import save_memory, search_memory
from rag import retrieve_from_rag


load_dotenv()


CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="basic"
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"
    


@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCX, TXT, notes, files, or documents.
    """

    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )




@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )



@tool
def recall_memory(query: str) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )


# ─── Weather Tool ─────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for any city in the world.
    Use this when the user asks about weather, temperature, climate, or conditions for a location.
    Input: City name like 'Lahore', 'New York', 'London', 'Tokyo'
    """
    try:
        api_key = os.getenv("WEATHERSTACK_API_KEY")
        if not api_key:
            return "Weather API key not configured. Please set WEATHERSTACK_API_KEY."

        url = "http://api.weatherstack.com/current"
        params = {
            "access_key": api_key,
            "query": city
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "error" in data:
            error_info = data["error"].get("info", "Unknown error")
            return f"Could not get weather for '{city}': {error_info}"

        current = data.get("current", {})
        location = data.get("location", {})

        loc_name = location.get("name", city)
        country = location.get("country", "")
        temp = current.get("temperature", "N/A")
        descriptions = ", ".join(current.get("weather_descriptions", ["N/A"]))
        humidity = current.get("humidity", "N/A")
        wind_speed = current.get("wind_speed", "N/A")
        wind_dir = current.get("wind_dir", "")
        feels_like = current.get("feelslike", "N/A")
        uv_index = current.get("uv_index", "N/A")
        visibility = current.get("visibility", "N/A")

        return (
            f"🌍 Weather in {loc_name}, {country}:\n"
            f"🌡️ Temperature: {temp}°C\n"
            f"🌤️ Condition: {descriptions}\n"
            f"🌡️ Feels Like: {feels_like}°C\n"
            f"💧 Humidity: {humidity}%\n"
            f"💨 Wind: {wind_speed} km/h {wind_dir}\n"
            f"👁️ Visibility: {visibility} km\n"
            f"☀️ UV Index: {uv_index}"
        )

    except requests.exceptions.Timeout:
        return f"Weather request timed out for '{city}'. Please try again."
    except Exception as e:
        return f"Error fetching weather for '{city}': {str(e)}"


# ─── Stock Price Tool ─────────────────────────────────────────

@tool
def get_stock_price(symbol: str) -> str:
    """
    Get current stock price and market data for any stock ticker.
    Use this when the user asks about stock prices, shares, market data, or financial info.
    Input: Stock ticker symbol like AAPL, GOOGL, MSFT, TSLA, AMZN, META, NVDA etc.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol.upper().strip())

        # Try to get info
        try:
            info = ticker.info
        except Exception:
            info = {}

        name = info.get("shortName", symbol.upper())
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
        market_cap = info.get("marketCap")
        day_high = info.get("dayHigh")
        day_low = info.get("dayLow")
        volume = info.get("volume")
        pe_ratio = info.get("trailingPE")
        week_52_high = info.get("fiftyTwoWeekHigh")
        week_52_low = info.get("fiftyTwoWeekLow")

        # Fallback to fast_info
        if price is None:
            try:
                fast = ticker.fast_info
                price = getattr(fast, "last_price", None)
                prev_close = getattr(fast, "previous_close", prev_close)
                market_cap = getattr(fast, "market_cap", market_cap)
            except Exception:
                pass

        if price is None:
            return f"Could not find stock data for '{symbol}'. Please check the ticker symbol."

        # Format change
        change_str = ""
        if prev_close and price:
            diff = price - prev_close
            pct = (diff / prev_close) * 100
            arrow = "📈" if diff >= 0 else "📉"
            change_str = f"{arrow} Change: ${diff:+.2f} ({pct:+.2f}%)"

        # Format market cap
        mc_str = "N/A"
        if market_cap:
            if market_cap >= 1e12:
                mc_str = f"${market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9:
                mc_str = f"${market_cap / 1e9:.2f}B"
            elif market_cap >= 1e6:
                mc_str = f"${market_cap / 1e6:.2f}M"
            else:
                mc_str = f"${market_cap:,.0f}"

        # Format volume
        vol_str = f"{volume:,}" if volume else "N/A"

        result = f"📊 {name} ({symbol.upper()})\n"
        result += f"💰 Price: ${price:.2f}\n"
        if change_str:
            result += f"{change_str}\n"
        if day_high:
            result += f"⬆️ Day High: ${day_high:.2f}\n"
        if day_low:
            result += f"⬇️ Day Low: ${day_low:.2f}\n"
        result += f"📦 Volume: {vol_str}\n"
        result += f"🏢 Market Cap: {mc_str}\n"
        if pe_ratio:
            result += f"📐 P/E Ratio: {pe_ratio:.2f}\n"
        if week_52_high:
            result += f"🔝 52W High: ${week_52_high:.2f}\n"
        if week_52_low:
            result += f"🔻 52W Low: ${week_52_low:.2f}"

        return result.strip()

    except Exception as e:
        return f"Error fetching stock data for '{symbol}': {str(e)}"



# ─── HITL: Purchase Stock ─────────────────────────────────────

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.
    This will ask for human confirmation before proceeding.
    Use when the user wants to BUY or PURCHASE stocks.
    """
    from langgraph.types import interrupt

    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined.",
            "symbol": symbol,
            "quantity": quantity,
        }


tools = [
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    get_weather,
    get_stock_price,
    purchase_stock,
    web_search
]