from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import requests





from dotenv import load_dotenv
import os

load_dotenv()


# --------------------------- CONFIG ---------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN not found in .env file")


##TELEGRAM_TOKEN = #Paste here your telegram_token#

# --------------------------- Memory Helpers ---------------------------
def get_memory_file(username):
    return f"{username}_chat_memory.json"

def load_memory(username):
    filename = get_memory_file(username)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

def save_memory(username, memory_list):
    filename = get_memory_file(username)
    with open(filename, "w") as f:
        json.dump(memory_list, f)

def build_recent_prompt(full_chat_history):
    # only last 10 exchanges = 20 messages
    recent = full_chat_history[-20:]
    messages = [("system", "You are a helpful assistant powered by Groq. Answer clearly.")]
    for m in recent:
        messages.append((m["role"], m["content"]))
    return ChatPromptTemplate.from_messages(messages)

# --------------------------- FastAPI MAIN APP ---------------------------
app = FastAPI()

# --------------------------- TEXT CHATBOT ENDPOINT ---------------------------
class ChatRequest(BaseModel):
    username: str
    message: str
    api_key: str | None = None

@app.post("/chat")
async def chat(req: ChatRequest):
    username = req.username
    message = req.message
    groq_api_key = req.api_key or os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return {"error": "Provide GROQ_API_KEY"}

    history = load_memory(username)
    history.append({"role": "user", "content": message})

    prompt = build_recent_prompt(history)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=DEFAULT_MODEL, temperature=0.7, streaming=False)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"question": message})
    history.append({"role": "assistant", "content": answer})
    save_memory(username, history)

    return {"answer": answer}

# --------------------------- TELEGRAM WEBHOOK ---------------------------
def send_telegram_msg(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})
# ---------- Typing helper ----------
def send_typing(chat_id):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendChatAction"
    requests.post(url, json={"chat_id": chat_id, "action": "typing"})

@app.post("/telegram-webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    chat_id = data["message"]["chat"]["id"]
    text = data["message"]["text"]

    # START command greeting
    if text == "/start":
        start_msg = "👋 Hello! I’m your LangChain + Groq AI assistant. Ask me anything!"
        send_telegram_msg(chat_id, start_msg)
        return {"ok": True}

    # Show typing action
    send_typing(chat_id)

    # Use chat_id as memory username
    result = await chat(ChatRequest(username=str(chat_id), message=text, api_key=None))
    bot_reply = result.get("answer", result.get("error", "Sorry, something went wrong."))
    send_telegram_msg(chat_id, bot_reply)
    return {"ok": True}
