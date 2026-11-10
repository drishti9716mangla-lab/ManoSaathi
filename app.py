import os
import base64
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
    "मर जाना चाहती", "खुद को नुकसान", "जीना नहीं", "खुदकुशी",
    "don't want to live", "dont want to live", "end it all", "suicide karna", "jeena nahi chahta", "jeena nahi chahti", "mar jana chahta", "mar jana chahti"
]
EMERGENCY_TERMS = [
    "can't breathe", "cannot breathe", "difficulty breathing", "severe chest pain",
    "unconscious", "not responding", "heavy bleeding", "सांस नहीं", "सांस लेने में कठिनाई",
    "सीने में तेज दर्द", "बेहोश", "बहुत खून बह",
    "chest pain", "chest pressure", "seizure", "took poison", "swallowed poison", "सीने में दर्द", "सीने में दबाव", "seene me dard", "saans nahi aa rahi", "zeher kha"
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


def crisis_kind(normalized_message: str):
    mental = any(normalize(t) in normalized_message for t in CRISIS_TERMS)
    physical = any(normalize(t) in normalized_message for t in EMERGENCY_TERMS)
    if mental and physical:
        return "both"
    return "mental" if mental else ("physical" if physical else None)

def comfort_reply(kind: str, language: str) -> str:
    if language == "hi":
        return {
            "mental": "मुझे बहुत दुख है कि आप ऐसा महसूस कर रहे हैं। आपको यह अकेले नहीं सहना है।",
            "physical": "यह गंभीर लग रहा है। कृपया अभी मदद लें।",
            "both": "मुझे बहुत दुख है कि आप इससे गुज़र रहे हैं। आप जो बता रहे हैं वह गंभीर हो सकता है, इसलिए कृपया अभी मदद लें। आपको यह अकेले नहीं सहना है।",
        }[kind]
    return {
        "mental": "I'm really sorry you're feeling this way. You don't have to go through this alone.",
        "physical": "That sounds serious. Please get help right now.",
        "both": "I'm really sorry you're going through this. What you describe can be serious, so please get help right now. You don't have to face this alone.",
    }[kind]

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


FAVICON_PNG = base64.b64decode("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")

@app.get("/favicon.ico")
def favicon():
    return Response(FAVICON_PNG, mimetype="image/png", headers={"Cache-Control": "public, max-age=86400"})

@app.get("/")
def home():
    # Serve the page as plain HTML (no Jinja templating), so characters like "{#" in CSS/JS can never break it.
    return Response((ROOT / "index.html").read_text(encoding="utf-8"), mimetype="text/html")

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
    kind = crisis_kind(normalized)
    if kind:
        headers["X-MS-Crisis"] = kind
        return Response(comfort_reply(kind, language), mimetype=mimetype, headers=headers)

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
