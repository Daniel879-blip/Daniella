# app.py
# Daniella – a friendly, smart AI assistant with Streamlit UI
# Works offline (rule-based + Wikipedia + SymPy), and can optionally use OpenAI if key is provided.
# Extras: image upload & analysis, chat memory, export, prompt suggestions, tools, and more.

import os
import io
import json
import time
import base64
import textwrap
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st

# --- Optional libs: best-effort imports (graceful fallback) ---
try:
    import sympy as sp
    SYMPY_OK = True
except Exception:
    SYMPY_OK = False

try:
    import wikipedia
    WIKI_OK = True
    wikipedia.set_lang("en")
except Exception:
    WIKI_OK = False

try:
    from PIL import Image, ImageStat, ExifTags
    PIL_OK = True
except Exception:
    PIL_OK = False

# Optional OCR if user installed pytesseract
try:
    import pytesseract
    OCR_OK = True
except Exception:
    OCR_OK = False

# Optional captioning if user installed transformers + torch
CAPTION_OK = False
try:
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    import torch  # noqa
    CAPTION_OK = True
except Exception:
    CAPTION_OK = False

# Optional OpenAI if user provided key
OPENAI_OK = False
_openai_client = None
if os.getenv("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
        OPENAI_OK = True
    except Exception:
        OPENAI_OK = False


APP_NAME = "Daniella"
APP_TAGLINE = "Your friendly, clever companion ✨"
DEFAULT_SYSTEM_PROMPT = (
    "You are Daniella: a warm, upbeat, extremely helpful AI assistant. "
    "Answer clearly, be concise, explain step-by-step when useful, and be respectful. "
    "If you don't know, say so briefly and suggest a next step."
)

# ---------------------- Helper: Session State ----------------------
def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.5
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 600
    if "use_openai" not in st.session_state:
        st.session_state.use_openai = OPENAI_OK
    if "persona" not in st.session_state:
        st.session_state.persona = "Friendly & Smart"
    if "image_model_ready" not in st.session_state:
        st.session_state.image_model_ready = False
    if "caption_model" not in st.session_state:
        st.session_state.caption_model = None
    if "caption_processor" not in st.session_state:
        st.session_state.caption_processor = None
    if "caption_tokenizer" not in st.session_state:
        st.session_state.caption_tokenizer = None


# ---------------------- Helper: Tools ----------------------
def tool_calculate_math(expr: str) -> Optional[str]:
    """Evaluate a math expression safely via sympy (if available)."""
    if not SYMPY_OK:
        return None
    try:
        # basic guard: limit length
        if len(expr) > 200:
            return "Expression too long to evaluate safely."
        sym_expr = sp.sympify(expr)
        simplified = sp.simplify(sym_expr)
        return f"Result: {simplified}"
    except Exception as e:
        return f"Could not evaluate expression: {e}"

def tool_wikipedia(query: str) -> Optional[str]:
    """Quick Wikipedia lookup (summary)."""
    if not WIKI_OK:
        return None
    try:
        results = wikipedia.search(query, results=1)
        if not results:
            return "I couldn't find a relevant Wikipedia page."
        page = wikipedia.page(results[0], auto_suggest=False, redirect=True)
        summary = wikipedia.summary(page.title, sentences=5, auto_suggest=False, redirect=True)
        return f"**{page.title}**\n\n{summary}"
    except Exception as e:
        return f"Wikipedia lookup error: {e}"

def tool_units_convert(value: float, from_unit: str, to_unit: str) -> Optional[str]:
    """Tiny unit converter demo (extensible)."""
    # Simple subset for demo (you can expand easily)
    # Distances (m, km, mi), Weights (g, kg, lb)
    try:
        from_unit = from_unit.strip().lower()
        to_unit = to_unit.strip().lower()

        # Distance
        dist_factors = {"m": 1.0, "km": 1000.0, "mi": 1609.344}
        if from_unit in dist_factors and to_unit in dist_factors:
            meters = value * dist_factors[from_unit]
            converted = meters / dist_factors[to_unit]
            return f"{value} {from_unit} = {converted:.6g} {to_unit}"

        # Weight
        w_factors = {"g": 1.0, "kg": 1000.0, "lb": 453.59237}
        if from_unit in w_factors and to_unit in w_factors:
            grams = value * w_factors[from_unit]
            converted = grams / w_factors[to_unit]
            return f"{value} {from_unit} = {converted:.6g} {to_unit}"

        return "Unsupported unit conversion (you can extend the table in code)."
    except Exception as e:
        return f"Unit conversion error: {e}"

def image_basic_info(img: Image.Image) -> Dict[str, Any]:
    """Compute basic stats and EXIF if available."""
    info = {}
    try:
        info["size"] = img.size
        info["mode"] = img.mode
        stat = ImageStat.Stat(img.convert("RGB"))
        info["mean_rgb"] = tuple(int(v) for v in stat.mean)  # average color
        info["extrema_rgb"] = stat.extrema
    except Exception:
        pass

    # EXIF
    try:
        exif = img._getexif()
        if exif:
            labeled = {}
            for k, v in exif.items():
                label = ExifTags.TAGS.get(k, k)
                labeled[label] = v
            info["exif"] = labeled
    except Exception:
        pass
    return info

def image_ocr(img: Image.Image) -> Optional[str]:
    if not OCR_OK:
        return None
    try:
        text = pytesseract.image_to_string(img)
        text = text.strip()
        return text if text else "(No text detected.)"
    except Exception as e:
        return f"OCR error: {e}"

# -------- Optional Image Captioning (heavy; loads on demand) --------
def ensure_caption_model():
    if not CAPTION_OK:
        return False
    if st.session_state.image_model_ready:
        return True
    try:
        with st.spinner("Loading image captioning model... (first time can be slow)"):
            model_id = "nlpconnect/vit-gpt2-image-captioning"
            model = VisionEncoderDecoderModel.from_pretrained(model_id)
            processor = ViTImageProcessor.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            st.session_state.caption_model = model
            st.session_state.caption_processor = processor
            st.session_state.caption_tokenizer = tokenizer
            st.session_state.image_model_ready = True
        return True
    except Exception:
        return False

def image_caption(img: Image.Image) -> Optional[str]:
    if not ensure_caption_model():
        return None
    try:
        model = st.session_state.caption_model
        processor = st.session_state.caption_processor
        tokenizer = st.session_state.caption_tokenizer
        pixel_values = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values
        output_ids = model.generate(pixel_values, max_length=20, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        return f"Captioning error: {e}"

# ---------------------- AI Backends ----------------------
def ai_answer_with_openai(messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(OpenAI error) {e}"

def offline_answer(user_text: str) -> str:
    """
    Lightweight offline reasoning with heuristics + tools.
    """
    text = user_text.strip()

    # Commands
    if text.startswith("/help"):
        return (
            "Commands:\n"
            "• /help — show this help\n"
            "• /math <expr> — evaluate math (e.g., /math (2+3)^4)\n"
            "• /wiki <topic> — quick Wikipedia summary (offline module)\n"
            "• /convert <value> <from> to <to> — unit convert\n"
            "• /time — current date/time\n"
            "• /clear — clear chat"
        )
    if text.startswith("/clear"):
        st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        return "History cleared ✅"
    if text.startswith("/time"):
        return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    if text.startswith("/math"):
        expr = text[len("/math"):].strip()
        if not expr:
            return "Usage: /math <expression>"
        res = tool_calculate_math(expr)
        return res or "SymPy not installed. Install with: pip install sympy"

    if text.startswith("/wiki"):
        q = text[len("/wiki"):].strip()
        if not q:
            return "Usage: /wiki <topic>"
        res = tool_wikipedia(q)
        if res:
            return res
        return "Wikipedia module not installed. Try: pip install wikipedia"

    if text.startswith("/convert"):
        # naive parse: "/convert 12 km to mi"
        try:
            parts = text.split()
            i_to = parts.index("to")
            value = float(parts[1])
            from_u = parts[2]
            to_u = parts[i_to + 1]
            res = tool_units_convert(value, from_u, to_u)
            return res or "Conversion failed."
        except Exception:
            return "Usage: /convert <value> <from_unit> to <to_unit> (e.g., /convert 12 km to mi)"

    # Heuristic routes
    # 1) Math-y content
    if any(k in text for k in ["integrate", "differentiate", "solve", "derivative", "limit", "factor", "expand"]) and SYMPY_OK:
        try:
            _ = sp.sympify(text)
            return tool_calculate_math(text) or "I tried, but couldn't evaluate that."
        except Exception:
            pass

    # 2) Quick wiki for "what is", "who is"
    lowered = text.lower()
    if WIKI_OK and (lowered.startswith(("what is", "who is", "tell me about", "define ")) or len(text.split()) <= 6):
        wiki = tool_wikipedia(text)
        if wiki and "error" not in wiki.lower():
            return wiki

    # 3) Default helpful answer
    return (
        "Here’s my take:\n\n"
        + "• I’m running fully offline right now, so I’ll reason from general knowledge and simple tools.\n"
        + "• For deeper web-verified answers, add an OpenAI key in the sidebar.\n\n"
        + "Now, to your question:\n"
        + _generic_helpful_response(text)
    )

def _generic_helpful_response(text: str) -> str:
    # Short, friendly, structured response skeleton
    return (
        f"**Understanding**: You're asking about “{text}”.\n\n"
        "**Quick answer**: I’d approach this by breaking it into key points, giving a concise explanation, "
        "then next steps if you want to explore further.\n\n"
        "**Next steps**: Tell me the exact angle you care about, and I’ll tailor the answer. "
        "You can also try `/wiki <topic>` or `/math <expr>`."
    )

def ai_answer(user_text: str, temperature: float, max_tokens: int, use_openai: bool) -> str:
    if use_openai and OPENAI_OK:
        # Build chat messages excluding system for OpenAI
        msgs = [m for m in st.session_state.messages if m["role"] in ("system", "user", "assistant")]
        return ai_answer_with_openai(msgs, temperature, max_tokens)
    else:
        return offline_answer(user_text)


# ---------------------- UI Utilities ----------------------
def avatar_for(role: str) -> str:
    if role == "user":
        return "👤"
    if role == "assistant":
        return "🤖"
    return "⚙️"

def download_button_bytes(filename: str, data: bytes, label: str):
    b64 = base64.b64encode(data).decode()
    href = f'<a download="{filename}" href="data:application/octet-stream;base64,{b64}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def export_chat_md(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        if m["role"] == "system":
            continue
        who = "You" if m["role"] == "user" else APP_NAME
        parts.append(f"### {who}\n\n{m['content']}\n")
    return "\n".join(parts)

# ---------------------- Layout ----------------------
st.set_page_config(
    page_title=f"{APP_NAME} – {APP_TAGLINE}",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_state()

with st.sidebar:
    st.markdown(f"## {APP_NAME} Settings")
    st.caption(APP_TAGLINE)

    st.session_state.persona = st.selectbox(
        "Persona",
        ["Friendly & Smart", "Teacher Mode (step-by-step)", "Concise & Direct", "Creative & Playful"],
        index=["Friendly & Smart","Teacher Mode (step-by-step)","Concise & Direct","Creative & Playful"].index(st.session_state.persona)
    )

    st.session_state.temperature = st.slider("Creativity (temperature)", 0.0, 1.0, st.session_state.temperature, 0.05)
    st.session_state.max_tokens = st.slider("Max tokens per reply", 128, 2000, st.session_state.max_tokens, 32)

    # Toggle OpenAI usage
    st.session_state.use_openai = st.toggle("Use OpenAI if available", value=st.session_state.use_openai, help="Requires OPENAI_API_KEY in environment.")
    if not OPENAI_OK:
        st.info("No OpenAI key detected. Add env var OPENAI_API_KEY to enable cloud brain.")

    st.markdown("---")
    st.markdown("### Quick Actions")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
            st.rerun()
    with colB:
        if st.button("Insert /help", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "/help"})
            st.rerun()

    st.markdown("---")
    st.markdown("### Export")
    md = export_chat_md(st.session_state.messages)
    download_button_bytes("Daniella_chat.md", md.encode("utf-8"), "⬇️ Download chat (Markdown)")
    st.caption("Tip: You can also copy-paste directly.")

    st.markdown("---")
    st.markdown("### Status")
    st.write(f"SymPy: {'✅' if SYMPY_OK else '—'}  |  Wikipedia: {'✅' if WIKI_OK else '—'}")
    st.write(f"OCR: {'✅' if OCR_OK else '—'}  |  Captioning model: {'✅' if CAPTION_OK else '—'}")
    st.write(f"OpenAI: {'✅' if OPENAI_OK else '—'}  |  PIL: {'✅' if PIL_OK else '—'}")


st.title(f"{APP_NAME} ✨")
st.caption(APP_TAGLINE)

tabs = st.tabs(["💬 Chat", "🖼️ Image Lab", "🧰 Tools", "ℹ️ About"])

# ---------------------- Tab: Chat ----------------------
with tabs[0]:
    # Chat history render
    for m in st.session_state.messages:
        if m["role"] == "system":
            continue
        with st.chat_message(m["role"], avatar=avatar_for(m["role"])):
            st.markdown(m["content"])

    # Smart suggestions row
    st.markdown("###### Suggestions:")
    sug1, sug2, sug3, sug4 = st.columns(4)
    with sug1: st.button("What is machine learning?", key="s1", on_click=lambda: st.session_state.messages.append({"role":"user","content":"What is machine learning?"}))
    with sug2: st.button("Summarize the Roman Empire", key="s2", on_click=lambda: st.session_state.messages.append({"role":"user","content":"Summarize the Roman Empire in 5 bullet points."}))
    with sug3: st.button("Explain gradients simply", key="s3", on_click=lambda: st.session_state.messages.append({"role":"user","content":"Explain gradients like I'm 12."}))
    with sug4: st.button("Convert 12 km to mi", key="s4", on_click=lambda: st.session_state.messages.append({"role":"user","content":"/convert 12 km to mi"}))

    # Persona hint
    persona_instructions = {
        "Friendly & Smart": "Be warm, concise, and practical.",
        "Teacher Mode (step-by-step)": "Explain step-by-step, check understanding, give simple examples.",
        "Concise & Direct": "Answer directly in 4-7 crisp sentences.",
        "Creative & Playful": "Light humor, analogies, and vivid but clear language."
    }[st.session_state.persona]

    prompt = st.chat_input("Ask me anything… (try /help)")
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Compose system persona layer
        system_msg = {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT + f" Persona: {st.session_state.persona}. Guideline: {persona_instructions}"
        }

        # For OpenAI calls, we keep a system at the start; for offline we just pass text
        if st.session_state.use_openai and OPENAI_OK:
            temp_msgs = [system_msg] + [m for m in st.session_state.messages if m["role"] != "system"]
        else:
            temp_msgs = [m for m in st.session_state.messages if m["role"] != "system"]

        with st.chat_message("assistant", avatar=avatar_for("assistant")):
            with st.spinner("Daniella is thinking…"):
                try:
                    reply = ai_answer(
                        user_text=prompt,
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens,
                        use_openai=st.session_state.use_openai,
                    )
                except Exception as e:
                    reply = f"Oops, I hit a snag:\n\n```\n{traceback.format_exc()}\n```"

            # Show and store reply
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------------------- Tab: Image Lab ----------------------
with tabs[1]:
    st.subheader("Upload an image for analysis")
    st.caption("Daniella can show metadata, average colors, OCR text, and optional captions (if transformers are installed).")

    uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    col1, col2 = st.columns([1, 1.2])
    if uploaded and PIL_OK:
        img = Image.open(uploaded)
        with col1:
            st.image(img, caption="Your image", use_container_width=True)
        with col2:
            st.markdown("##### Analysis")
            info = image_basic_info(img)
            st.json(info)

            if OCR_OK:
                with st.expander("🧾 OCR (text extraction)"):
                    with st.spinner("Reading text…"):
                        text = image_ocr(img)
                    st.code(text or "(No text)")

            if CAPTION_OK:
                with st.expander("📝 Caption (AI-generated)"):
                    if st.button("Generate caption"):
                        with st.spinner("Captioning…"):
                            cap = image_caption(img)
                        st.success(cap or "No caption.")
            else:
                st.info("Install transformers + torch to enable auto-captioning.")

    elif uploaded and not PIL_OK:
        st.error("Pillow (PIL) not installed. Try: pip install pillow")

    st.markdown("---")
    st.markdown("##### Tips")
    st.markdown(
        "- OCR works best with clear, high-contrast text.\n"
        "- Captions require `transformers` and can be slow the first time.\n"
        "- You can ask Daniella about the image content in the Chat tab too!"
    )

# ---------------------- Tab: Tools ----------------------
with tabs[2]:
    st.subheader("Quick Utilities")

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Math Evaluator**")
        expr = st.text_input("Expression (SymPy)", placeholder="(2+3)**5 / sqrt(2)")
        if st.button("Compute"):
            res = tool_calculate_math(expr) if expr else "Enter an expression."
            st.code(res or "SymPy not installed.")

    with t2:
        st.markdown("**Wikipedia Summary**")
        q = st.text_input("Topic", placeholder="Alan Turing")
        if st.button("Lookup"):
            res = tool_wikipedia(q) if q else "Enter a topic."
            st.markdown(res or "Wikipedia module not installed.")

    with t3:
        st.markdown("**Unit Converter**")
        colu = st.columns([1,1,1])
        with colu[0]:
            val = st.number_input("Value", value=1.0)
        with colu[1]:
            u_from = st.text_input("From", value="km")
        with colu[2]:
            u_to = st.text_input("To", value="mi")
        if st.button("Convert"):
            st.code(tool_units_convert(val, u_from, u_to))

# ---------------------- Tab: About ----------------------
with tabs[3]:
    st.markdown(f"### Meet {APP_NAME} 🤖")
    st.markdown(
        "Daniella is a friendly AI that can chat, analyze images, do math, look up topics on Wikipedia, "
        "and export your conversations. She works offline by default and can use cloud AI if you provide an API key."
    )
    st.markdown("**Commands:** `/help`, `/math <expr>`, `/wiki <topic>`, `/convert <v> <from> to <to>`, `/time`, `/clear`")
    st.markdown("**Privacy:** Your chat stays in your browser session unless you export it.")
    st.caption("Built with Streamlit. Extend freely—add tools, memory, or your own models!")
