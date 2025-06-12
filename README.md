# Resume Matcher using BERT and FastAPI

This is a web application that matches resumes with job descriptions using a BERT model and is enhanced by PyTorch. It supports both plain text and PDF resume inputs.

Features
- Resume vs Job Description matching
- PDF upload support
- Built with FastAPI, BERT, PyTorch
- Simple HTML frontend

Tech Stack
- Python 3
- FastAPI
- HuggingFace Transformers
- PyTorch
- PyMuPDF (for PDF parsing)

Installation Instructions
git clone https://github.com/yourusername/resume-matcher.git
cd resume-matcher
pip install -r requirements.txt

How to Run
# Start backend
cd backend
uvicorn app:app --reload

# Open in browser
http://localhost:8000

Project Structure
resume_classifier/
├── backend/
│   ├── app.py
│   ├── model/
│   │   └── model.py
│   └── static/
│       ├── index.html
│       └── script.js
├── train_model.py
├── resume.csv

