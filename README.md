# 🧠 factorial24

This repository contains solutions to two AI-based engineering and document analysis challenges:

1. **Resume Ranking System**: Ranks resumes based on relevance to a job description.
2. **Engineering Drawing Analyzer**: Extracts object dimensions from engineering drawings, annotates them, and generates structured Excel summaries.

---

# 🔍 AI Resume Ranking System

This repository contains an AI agent that ranks resumes based on relevance to a given job description using natural language processing, semantic similarity, and entity extraction.

## 🚀 Features

- 📄 PDF text extraction from resume files using `PyMuPDF`
- 🔍 NLP preprocessing using `spaCy`
- 🤖 Skill extraction via `BERT-based Named Entity Recognition`
- 🧠 Semantic similarity scoring using `Sentence Transformers`
- 📊 Multi-factor scoring combining similarity, skills, experience, and education
- 🔗 Section clustering for contextual understanding

## 🛠️ Technology Stack

- PyMuPDF
- spaCy
- Transformers (dslim/bert-base-NER)
- Sentence Transformers (MiniLM-L6-v2)
- scikit-learn
- pandas, NumPy

## 📁 Project Structure

```
Resume_Ranking/
├── Ranking.ipynb
└── data/
    ├── resume-01.pdf
    └── ...
requirements.txt
README.md
```

## ⚙️ Setup & Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 🧪 Usage

```bash
jupyter notebook Main/Ranking.ipynb
```

Define your job description inside the notebook and place resumes in `Main/data/`.

## 🔧 Scoring Configuration

```python
score = alpha * semantic_score + beta * skill_score + gamma * experience_score + delta * education_flag
```

Default Weights:

| alpha | beta | gamma | delta |
|-------|------|-------|--------|
| 0.5   | 0.3  | 0.1   | 0.1    |

---

# 📐 AI Engineering Drawing Analyzer

This AI pipeline processes PDF engineering drawings to extract and annotate dimensional data.

## ⚙️ Workflow

1. **PDF Input** → `engineering_drawings/`
2. **PDF to Images** using `pdf2image`
3. **OCR + Shape Detection** using `pytesseract` and `OpenCV`
4. **Annotation** of detected shapes with bounding boxes and dimension labels
5. **Excel Report Generation** per drawing

## 📁 Project Structure

```
drawing_analyser/
├── engineering_drawings/     # Input PDFs
├── pdf_pages/                # Auto-generated images from PDF pages
├── annotated/                # Annotated images with dimension labels
├── output/                   # Excel summaries for each drawing
├── main.py                   # Main execution script
└── requirements.txt          # Python dependencies

```

## 🛠️ Tech Stack

- Python 3.8+
- OpenCV
- pdf2image
- pytesseract
- pandas

## 🔧 Setup Instructions

```bash
pip install -r requirements.txt
```

Tesseract OCR must be installed and added to PATH.

Windows: https://github.com/UB-Mannheim/tesseract/wiki  
Linux: `sudo apt install tesseract-ocr`

## 🚀 Run

```bash
python main.py
```

---
