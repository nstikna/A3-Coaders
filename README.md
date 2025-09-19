
# AI Medical Assistant

**Short description**

A web app that reads medical reports (PDF / image / raw text), extracts the text, and uses a fine-tuned T5 model to simplify the report into patient-friendly language. Frontend is a chat-like UI (Bootstrap + JS). Backend is Flask, with PDF/image extraction via `pdfplumber`, `PyMuPDF` (fitz) and Tesseract OCR as fallback.

---

## Table of contents

1. What it does  
2. Repo layout  
3. Requirements  
4. Installation & setup  
5. How to run  
6. API: `/simplify` endpoint  
7. Frontend (where files live)  
8. Model & training notes  
9. Improving extraction & accuracy  

---

## What it does

- Upload **PDF, JPG, PNG** or type text directly.  
- Extracts text from file using:  
  1. pdfplumber (primary)  
  2. PyMuPDF (fallback for scanned PDFs)  
  3. Tesseract OCR (fallback for images)  
- Simplifies technical reports with a **fine-tuned T5 model**.  
- Returns clean bullet-pointed explanations in simple language.  
- Chat-like interface where users can type questions and get simplified answers.  

---

## Repo layout

```
/static/          # Frontend HTML, CSS, JS (index.html, about.html, review.html, contact.html)
/uploads/         # Temporary uploaded files
medicalsimplifiermodel/  # Fine-tuned T5 model files
app.py            # Flask backend
README.md         # Project documentation
```

---

## Requirements

- Python 3.9+  
- Flask  
- flask-cors  
- transformers (HuggingFace)  
- torch (with CUDA if available)  
- pdfplumber  
- PyMuPDF (`fitz`)  
- pytesseract + Tesseract installed  
- pillow  

---

## Installation & setup

```bash
# Clone repo
git clone https://github.com/yourusername/ai-medical-assistant.git
cd ai-medical-assistant

# Create virtual env
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate    # (Windows)

# Install dependencies
pip install -r requirements.txt
```

Make sure **Tesseract** is installed:  
- Linux: `sudo apt install tesseract-ocr`  
- Mac: `brew install tesseract`  
- Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki  

---

## How to run

```bash
python app.py
```

- Flask server starts on **http://127.0.0.1:5002/**  
- Open `http://127.0.0.1:5002/` in browser to use frontend.  

---

## API: `/simplify` endpoint

### POST `/simplify`  

- **Input**  
  - JSON: `{ "text": "medical text" }`  
  - or multipart: `{ file: <pdf|jpg|png> }`  

- **Output**  
```json
{
  "simplified_text": "• Patient has fever • Symptoms suggest malaria"
}
```

---

## Frontend (where files live)

- `static/index.html` → main chat UI  
- `static/about.html` → about page  
- `static/review.html` → expert reviews  
- `static/contact.html` → contact form  

The **chat UI** simulates conversation between user and AI, styled with Bootstrap 5 + FontAwesome.  

---

## Model & training notes

- Model used: **T5ForConditionalGeneration** fine-tuned on **4000+ medical dataset samples**.  
- Task format: `"simplify in English: <text>" → "<simplified patient-friendly explanation>"`.  
- Inference: beam search with `num_beams=4`, `max_length=512`.  

---

## Improving extraction & accuracy

1. **Extraction:** If pdfplumber fails, PyMuPDF is used. If both fail, fallback to Tesseract OCR.  
2. **Formatting:** Model output is split into sentences, turned into bullet points.  
3. **Context memory:** Could add session history (cache last N messages per user).  
4. **Better UI:** Add context-follow-up so bot “remembers” last answer.  
5. **Future idea:** Connect to medical knowledge base (like UMLS) for structured answers.  

---

## Demo video script (3 min)

1. **Intro:** "This is AI Medical Assistant, a tool that simplifies medical reports."  
2. **Upload demo:** Upload a PDF → show extracted text → simplified output.  
3. **Image demo:** Upload a JPG scan → OCR → simplified text.  
4. **Text demo:** Type "malaria symptoms" → bot explains in layman terms.  
5. **Closing:** "This helps patients understand their reports in simple language."  

---

## License

MIT License © 2024 AI Medical Assistant Team
