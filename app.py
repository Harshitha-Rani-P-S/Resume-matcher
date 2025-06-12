from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import fitz  # PyMuPDF

from model.model import load_model_and_tokenizer, predict_match

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# Load model and tokenizer
model, tokenizer, device = load_model_and_tokenizer()

# Schema for text input
class ResumeInput(BaseModel):
    job_description: str
    resume_text: str

# Text input prediction
@app.post("/predict")
def get_prediction(data: ResumeInput):
    match, confidence = predict_match(data.job_description, data.resume_text, model, tokenizer, device)
    return {"match": match, "confidence": confidence}

# PDF input prediction
@app.post("/predict-pdf")
async def predict_from_pdf(resume_file: UploadFile = File(...), job_description: str = Form(...)):
    contents = await resume_file.read()

    text = ""
    with fitz.open(stream=contents, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    # Debug log to verify text quality
    print("=== Job Description ===")
    print(job_description)
    print("=== Extracted Resume Text ===")
    print(text[:500])  # print first 500 chars

    if not text.strip():
        return {"match": False, "confidence": 0.0, "error": "No text extracted from PDF"}

    match, confidence = predict_match(job_description, text, model, tokenizer, device)
    return {"match": match, "confidence": confidence}

