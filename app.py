# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import os
import io
import uuid
import time

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

# ------- LOAD MODEL (your trained model folder) -------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model = T5ForConditionalGeneration.from_pretrained("./medicalsimplifiermodel").to(device)
tokenizer = T5Tokenizer.from_pretrained("./medicalsimplifiermodel")

# ------- in-memory conversation store (session_id -> list of messages) -------
# each message: {"role":"user"|"assistant","text": "...", "ts": <timestamp>}
conversations = {}
MAX_HISTORY_TURNS = 6  # how many recent turns to include in context


# ------- PDF text extractor with OCR fallback -------
def extract_text_from_pdf(file_path):
    text = ""

    # First pass: pdfplumber, page by page. if page has no text -> OCR that page
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text += page_text + "\n"
                else:
                    # OCR fallback for that page
                    # convert page to image via PyMuPDF for better resolution & OCR
                    try:
                        with fitz.open(file_path) as doc:
                            pix = doc[i].get_pixmap(dpi=300)
                            img = Image.open(io.BytesIO(pix.tobytes("png")))
                            ocr_text = pytesseract.image_to_string(img)
                            text += ocr_text + "\n"
                    except Exception:
                        # if fitz not working for a page, attempt saving page image via pdfplumber
                        try:
                            im = page.to_image(resolution=300).original
                            ocr_text = pytesseract.image_to_string(im)
                            text += ocr_text + "\n"
                        except Exception:
                            pass
    except Exception as e:
        print("pdfplumber failed:", e)

    # If result is empty or very short, do full fallback with PyMuPDF + OCR
    if not text.strip() or len(text.strip()) < 80:
        try:
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    page_text = page.get_text("text") or ""
                    if page_text.strip():
                        text += page_text + "\n"
                    else:
                        # OCR fallback for page
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            print("PyMuPDF fallback failed:", e)

    return text.strip()


# ------- utilities -------
def truncate_for_prompt(s: str, max_chars=3000):
    """Keep the head and tail of long reports to preserve context."""
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    head = s[: max_chars // 2]
    tail = s[- max_chars // 2 :]
    return head + "\n\n...[truncated]...\n\n" + tail


def build_prompt_from_history(session_id, user_text, report_text=None):
    """Construct the prompt for the model by including a system instruction + recent turns."""
    system_instruction = (
        "You are a medical-report simplifier for patients. "
        "Do NOT repeat the instruction. Keep language simple and clear — use short sentences or bullet points. "
        "If the user asks a follow-up (e.g., 'what does it mean?'), refer to prior messages in this conversation and answer based on previously provided report fragments. "
        "If the report includes numeric values, mention only what is important for the patient (e.g., diagnosis, abnormal values) and avoid technical jargon. "
        "If uncertain, say you are unsure and recommend following up with clinician. "
        "Answer concisely and in patient-friendly terms."
    )

    chat_lines = [f"SYSTEM: {system_instruction}"]

    history = conversations.get(session_id, [])[-(MAX_HISTORY_TURNS * 2) :]  # last turns
    # format previous messages
    for m in history:
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        # limit length of each stored message to avoid too-long prompt
        content = (m["text"][:1500] + "..." ) if len(m["text"]) > 1500 else m["text"]
        chat_lines.append(f"{role}: {content}")

    # include a short (truncated) version of the report if provided
    if report_text:
        truncated_report = truncate_for_prompt(report_text, max_chars=3000)
        chat_lines.append("REPORT: " + truncated_report)

    # Finally the new user query
    chat_lines.append("USER: " + user_text.strip())
    chat_lines.append("ASSISTANT:")  # model should fill this

    return "\n\n".join(chat_lines)


def postprocess_model_output(s: str) -> str:
    """Clean up output, remove echo of the prompt, and format bullet points one per line."""
    if not s:
        return ""
    # remove leading instruction echoes like 'Simplify this...'
    s = s.replace("Simplify and explain this medical information clearly:", "")
    s = s.strip()

    # If the model returned everything in one blob, try to split into sentences and bullets.
    # Use periods and newlines to split, but keep numeric patterns together.
    lines = []
    # break on newline first
    for chunk in s.splitlines():
        chunk = chunk.strip()
        if not chunk:
            continue
        # if chunk contains multiple sentences, split by '. ' but preserve numeric decimals
        parts = [p.strip() for p in chunk.replace("•", ". ").split(". ") if p.strip()]
        for p in parts:
            if p and p.lower() not in (x.lower() for x in lines):
                lines.append(p)

    if len(lines) == 1:
        return lines[0]
    return "\n".join(f"• {l}" for l in lines)


# ------- main generator that uses prompt and model -------
def generate_simplified_response(session_id, user_text, report_text=None):
    prompt = build_prompt_from_history(session_id, user_text, report_text)

    # tokenize - truncation ensures prompt fits model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    # generate
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=6,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
    )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    processed = postprocess_model_output(raw)
    return processed


# ------- API endpoints -------
@app.route('/simplify', methods=['POST'])
def simplify():
    try:
        session_id = request.form.get("session_id") or request.args.get("session_id")
        # create new session if none provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # ensure conversation exists
        conversations.setdefault(session_id, [])

        # uploaded file?
        if "file" in request.files:
            f = request.files["file"]
            if not f or f.filename == "":
                return jsonify({"error": "No file uploaded"}), 400

            os.makedirs("./uploads", exist_ok=True)
            file_path = os.path.join("./uploads", f"{int(time.time())}_{f.filename}")
            f.save(file_path)

            ext = f.filename.lower()
            if ext.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file_path)
            elif ext.endswith((".jpg", ".jpeg", ".png")):
                img = Image.open(file_path)
                extracted_text = pytesseract.image_to_string(img)
            else:
                return jsonify({"error": "Unsupported file format"}), 400

            if not extracted_text.strip():
                return jsonify({"error": "No readable text found in file"}), 400

            # add user message (report upload) to conversation
            conversations[session_id].append({"role": "user", "text": f"[Uploaded report: {f.filename}]", "ts": time.time()})
            # ask model to summarize the report (user may have given additional message too)
            user_query = request.form.get("text") or "Please summarize this report for the patient."
            conversations[session_id].append({"role": "user", "text": user_query, "ts": time.time()})

            assistant_text = generate_simplified_response(session_id, user_query, report_text=extracted_text)
            # save assistant reply
            conversations[session_id].append({"role": "assistant", "text": assistant_text, "ts": time.time()})

            return jsonify({"simplified_text": assistant_text, "session_id": session_id})

        # plain JSON text
        data = request.get_json(silent=True) or {}
        user_text = data.get("text") or request.form.get("text") or ""
        if not user_text.strip():
            return jsonify({"error": "No text provided"}), 400

        # save user message
        conversations[session_id].append({"role": "user", "text": user_text, "ts": time.time()})

        # generate reply using history
        assistant_text = generate_simplified_response(session_id, user_text, report_text=None)

        # save assistant reply
        conversations[session_id].append({"role": "assistant", "text": assistant_text, "ts": time.time()})

        return jsonify({"simplified_text": assistant_text, "session_id": session_id})

    except Exception as e:
        # return the exception message for debugging (in prod, sanitize)
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_session():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if session_id and session_id in conversations:
        conversations.pop(session_id, None)
        return jsonify({"status": "reset"})
    return jsonify({"status": "no_session"}), 400


@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')


if __name__ == "__main__":
    # port 5002 by default
    app.run(host='0.0.0.0', port=5002, debug=True)