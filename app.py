import os
import re
import json
import csv
import random
import unicodedata
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, render_template
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

app = Flask(__name__, template_folder=".")
app.config["JSON_AS_ASCII"] = False

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# Optional: point to another OpenAI-compatible provider (e.g. Google Gemini or Groq).
BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "").strip()
# Optional backup provider (used automatically if the main one fails), e.g. Groq.
BACKUP_API_KEY = os.getenv("BACKUP_API_KEY", "").strip()
BACKUP_BASE_URL = os.getenv("BACKUP_BASE_URL", "").strip() or None
BACKUP_MODEL = os.getenv("BACKUP_MODEL", "").strip()
DATASET_PATH = Path(os.getenv("KAGGLE_DATASET_PATH", str(ROOT / "data" / "kaggle_dataset.csv")))
INTENTS_PATH = Path(os.getenv("INTENTS_PATH", str(ROOT / "data" / "intents.json")))

CRISIS_TERMS = [
    "suicide", "kill myself", "end my life", "want to die", "hurt myself",
    "self harm", "self-harm", "आत्महत्या", "खुद को मार", "मर जाना चाहता",
    "मर जाना चाहती", "खुद को नुकसान", "जीना नहीं", "खुदकुशी"
]
EMERGENCY_TERMS = [
    "can't breathe", "cannot breathe", "difficulty breathing", "severe chest pain",
    "unconscious", "not responding", "heavy bleeding", "सांस नहीं", "सांस लेने में कठिनाई",
    "सीने में तेज दर्द", "बेहोश", "बहुत खून बह"
]

def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def load_intents() -> list[dict]:
    if not INTENTS_PATH.exists():
        return []
    try:
        with INTENTS_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("intents", []) if isinstance(data, dict) else []
    except (OSError, json.JSONDecodeError):
        app.logger.exception("Could not load intents dataset")
        return []

INTENTS = load_intents()

def find_intent(message: str):
    text = normalize(message)
    best, best_score = None, 0.0
    tokens = set(text.split())
    for intent in INTENTS:
        for pattern in intent.get("patterns", []):
            p = normalize(str(pattern))
            if not p:
                continue
            if p in text:
                return intent
            pt = set(p.split())
            if pt:
                score = len(tokens & pt) / len(pt)
                if score > best_score:
                    best, best_score = intent, score
    return best if best_score >= 0.6 else None

def load_kaggle_rows(limit: int = 2500) -> list[dict[str, str]]:
    """Read CSV/JSON dataset if the user places it in data/ or configures its path."""
    if not DATASET_PATH.exists():
        return []
    try:
        if DATASET_PATH.suffix.lower() == ".csv":
            with DATASET_PATH.open(encoding="utf-8-sig", newline="") as f:
                return [dict(row) for _, row in zip(range(limit), csv.DictReader(f))]
        if DATASET_PATH.suffix.lower() == ".json":
            raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [x for x in raw[:limit] if isinstance(x, dict)]
            if isinstance(raw, dict):
                for key in ("data", "intents", "conversations", "records"):
                    if isinstance(raw.get(key), list):
                        return [x for x in raw[key][:limit] if isinstance(x, dict)]
        return []
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
        app.logger.exception("Could not read Kaggle dataset")
        return []

KAGGLE_ROWS = load_kaggle_rows()

def dataset_context(message: str, max_rows: int = 4) -> str:
    """Simple lexical retrieval. The dataset is reference material, not a diagnosis engine."""
    if not KAGGLE_ROWS:
        return ""
    query = set(normalize(message).split())
    scored = []
    for row in KAGGLE_ROWS:
        fields = [str(v) for v in row.values() if v is not None]
        joined = " ".join(fields)
        tokens = set(normalize(joined).split())
        overlap = len(query & tokens)
        if overlap:
            scored.append((overlap / max(len(query), 1), joined[:900]))
    scored.sort(key=lambda item: item[0], reverse=True)
    return "\n".join(f"- {text}" for _, text in scored[:max_rows])

def crisis_reply(language: str) -> str:
    if language == "hi":
        return ("मुझे दुख है कि आप इस कठिन समय से गुजर रहे हैं। अभी किसी भरोसेमंद व्यक्ति के साथ रहें और अकेले न रहें। "
                "यदि आपको खुद को नुकसान पहुँचाने का डर है, तो तुरंत 112 पर कॉल करें या नज़दीकी आपातकालीन विभाग जाएँ। "
                "भारत में मानसिक स्वास्थ्य सहायता के लिए टेली-मानस 14416 पर कॉल करें। मैं AI हूँ, आपातकालीन सेवा नहीं।")
    return ("I'm sorry you're going through this. Please stay with someone you trust and don't remain alone. "
            "If you may hurt yourself or are in immediate danger, call 112 in India or go to the nearest emergency department now. "
            "For mental-health support in India, call Tele-MANAS at 14416. I am an AI tool, not an emergency service.")

def emergency_reply(language: str) -> str:
    if language == "hi":
        return "यह आपातकाल हो सकता है। भारत में अभी 112 पर कॉल करें या तुरंत आपातकालीन चिकित्सा सहायता लें। चैटबॉट के उत्तर की प्रतीक्षा न करें।"
    return "This may be an emergency. Call India's emergency number 112 or seek emergency medical care now. Do not wait for a chatbot response."

def dataset_fallback(message: str, language: str) -> str:
    intent = find_intent(message)
    if intent and intent.get("responses"):
        return random.choice(intent["responses"])
    if language == "hi":
        return ("मैं आपकी बात समझना चाहता हूँ। क्या आप थोड़ा और बता सकते हैं कि अभी क्या महसूस हो रहा है? "
                "मैं सामान्य सहायता दे सकता हूँ, लेकिन निदान या पेशेवर उपचार का विकल्प नहीं हूँ।")
    return ("I want to understand what you're experiencing. Could you tell me a little more about what has been happening? "
            "I can offer general support, but I cannot diagnose or replace a qualified professional.")

def build_system_prompt(language: str, context: str) -> str:
    language_rule = "Respond entirely in natural Hindi (Devanagari)." if language == "hi" else "Respond in clear, natural English."
    dataset_rule = f"\nRelevant dataset reference snippets (may be incomplete; use only if relevant):\n{context}" if context else ""
    return (
        "You are ManoSaathi, a warm, non-judgmental wellbeing and general health information assistant for users in India. "
        "You are not a doctor, therapist, diagnostic system, or emergency service. "
        f"{language_rule} Match the user's language if they mix languages, while keeping the selected site language. "
        "Listen first, acknowledge feelings, ask one useful follow-up question when needed, and answer the actual latest message. "
        "Use prior conversation history for continuity. Give practical, low-risk wellbeing suggestions. "
        "For physical symptoms or illness (fever, cough, pain, vomiting, rash, etc.): first give a short, caring supportive sentence or two, "
        "then do NOT name medicines, doses or home cures. Instead tell the user to open the Symptom Screener on this website "
        "(the 'Check' tab, 'लक्षण जाँच' in Hindi) to organise their symptoms, and to see a qualified clinician if symptoms are severe, persistent or worsening. "
        "You may mention only very basic comfort such as resting and drinking water. Keep it short. "
        "For possible emergencies, tell the user to call 112 in India or seek urgent care. "
        "For suicide/self-harm risk, encourage immediate real-person support, staying with a trusted person, and calling 112; "
        "also mention Tele-MANAS 14416 in India. Never shame, minimise, or claim confidentiality beyond this website's actual data handling. "
        "Keep replies concise, supportive, and easy to understand. Do not pretend dataset snippets are verified medical advice."
        + dataset_rule
    )

def ask_ai(message: str, language: str, history: list[dict], context: str) -> str:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured in .env")
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "system", "content": build_system_prompt(language, context)}]
    for item in history[-10:]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": message[:4000]})
    import time
    # Providers are tried in order: main first, then the optional backup.
    providers = [(API_KEY, BASE_URL, [MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != MODEL else []))]
    if BACKUP_API_KEY and BACKUP_MODEL:
        providers.append((BACKUP_API_KEY, BACKUP_BASE_URL, [BACKUP_MODEL]))
    last_exc = None
    result = None
    for key, base, model_list in providers:
        provider_client = OpenAI(api_key=key, base_url=base, timeout=10, max_retries=0)
        for model_name in model_list:
            for attempt in range(2):
                try:
                    result = provider_client.chat.completions.create(
                        model=model_name, messages=messages, temperature=0.65, max_tokens=700)
                    break
                except Exception as exc:
                    last_exc = exc
                    app.logger.warning("Provider failed (%s, %s): %r | %s", base or "openai", model_name, exc, getattr(getattr(exc, "response", None), "text", ""))
                    if getattr(exc, "status_code", None) in (429, 500, 502, 503, 504) and attempt < 1:
                        time.sleep(1.5)
                        continue
                    break
            if result is not None:
                break
        if result is not None:
            break
    if result is None:
        raise last_exc
    return (result.choices[0].message.content or "").strip()


def stream_ai(message: str, language: str, history: list[dict], context: str):
    """Yield the reply in small pieces as the model writes it (providers tried in order)."""
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    from openai import OpenAI
    messages = [{"role": "system", "content": build_system_prompt(language, context)}]
    for item in history[-10:]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": message[:4000]})
    providers = [(API_KEY, BASE_URL, [MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != MODEL else []))]
    if BACKUP_API_KEY and BACKUP_MODEL:
        providers.append((BACKUP_API_KEY, BACKUP_BASE_URL, [BACKUP_MODEL]))
    last_exc = None
    for key, base, model_list in providers:
        client = OpenAI(api_key=key, base_url=base, timeout=10, max_retries=0)
        for model_name in model_list:
            started = False
            try:
                stream = client.chat.completions.create(
                    model=model_name, messages=messages, temperature=0.65, max_tokens=700, stream=True)
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    piece = getattr(chunk.choices[0].delta, "content", None)
                    if piece:
                        started = True
                        yield piece
                if started:
                    return
            except Exception as exc:
                last_exc = exc
                app.logger.warning("Provider failed (%s, %s): %r | %s", base or "openai", model_name, exc,
                                   getattr(getattr(exc, "response", None), "text", ""))
                if started:      # already sent part of the reply; cannot switch provider mid-answer
                    return
    raise last_exc or RuntimeError("No AI provider produced a reply")

@app.after_request
def add_cors(resp):
    # Lets the page work when opened from Live Server / another local port.
    origin = os.getenv("CORS_ORIGIN", "").strip()
    if not origin:
        return resp
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

@app.route("/api/chat", methods=["OPTIONS"])
def chat_preflight():
    return ("", 204)

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "ai_configured": bool(API_KEY),
        "model": MODEL if API_KEY else None,
        "kaggle_dataset_loaded": bool(KAGGLE_ROWS),
        "kaggle_rows_loaded": len(KAGGLE_ROWS),
        "intent_count": len(INTENTS),
    })


@app.post("/api/chat_stream")
def chat_stream():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    language = "hi" if str(payload.get("language", "en")).lower().startswith("hi") else "en"
    history = payload.get("history", [])
    if not message:
        return jsonify({"error": "Please type a message first."}), 400
    if not isinstance(history, list):
        history = []

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    mimetype = "text/plain"
    normalized = normalize(message)
    if any(normalize(term) in normalized for term in CRISIS_TERMS):
        return Response(crisis_reply(language), mimetype=mimetype, headers=headers)
    if any(normalize(term) in normalized for term in EMERGENCY_TERMS):
        return Response(emergency_reply(language), mimetype=mimetype, headers=headers)

    context = dataset_context(message)

    def generate():
        got_any = False
        try:
            for piece in stream_ai(message, language, history, context):
                got_any = True
                yield piece
        except Exception as exc:
            app.logger.warning("AI stream unavailable: %r", exc)
        if not got_any:
            reply = dataset_fallback(message, language)
            if language == "hi" and reply and not re.search(r"[\u0900-\u097f]", reply):
                reply = "अभी AI सेवा उपलब्ध नहीं है। " + reply
            yield reply

    return Response(generate(), mimetype=mimetype, headers=headers)

@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    language = "hi" if str(payload.get("language", "en")).lower().startswith("hi") else "en"
    history = payload.get("history", [])
    if not message:
        return jsonify({"error": "Please type a message first."}), 400
    if not isinstance(history, list):
        history = []

    normalized = normalize(message)
    if any(normalize(term) in normalized for term in CRISIS_TERMS):
        return jsonify({"reply": crisis_reply(language), "source": "safety", "matched": False})
    if any(normalize(term) in normalized for term in EMERGENCY_TERMS):
        return jsonify({"reply": emergency_reply(language), "source": "safety", "matched": False})

    context = dataset_context(message)
    try:
        reply = ask_ai(message, language, history, context)
        if not reply:
            raise RuntimeError("The AI provider returned an empty response.")
        return jsonify({"reply": reply, "source": "ai_api", "dataset_context_used": bool(context)})
    except Exception as exc:
        app.logger.warning("AI response unavailable: %r | body=%s | model=%s | base_url=%s",
                           exc, getattr(getattr(exc, "response", None), "text", ""), MODEL, BASE_URL)
        # Use the intent dataset only as a transparent offline fallback.
        reply = dataset_fallback(message, language)
        if language == "hi" and reply and not re.search(r"[\u0900-\u097f]", reply):
            reply = "अभी AI सेवा उपलब्ध नहीं है। " + reply
        return jsonify({
            "reply": reply,
            "source": "dataset_fallback" if INTENTS else "fallback",
            "warning": "AI API unavailable; configure OPENAI_API_KEY and check your connection."
        })

if __name__ == "__main__":
    import threading, webbrowser
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    # Open the site automatically (only in the main process, not the debug reloader child).
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\nManoSaathi running -> http://127.0.0.1:{port}")
    print(f"  .env found : {(ROOT / '.env').exists()}")
    print(f"  key loaded : {bool(API_KEY)}")
    print(f"  base URL   : {BASE_URL or '(OpenAI default)'}")
    print(f"  model      : {MODEL}")
    print(f"  backup     : {BACKUP_MODEL if BACKUP_API_KEY else '(none)'}\n")
    app.run(host="127.0.0.1", port=port, debug=debug)
