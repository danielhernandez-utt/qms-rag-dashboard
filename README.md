---
title: Trustworthy AI SGC Assistant
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# Module 17: Trustworthy AI Production Monitoring Dashboard  
## Team Project – Multilingual RAG System with Explainability  

This project presents a **Trustworthy AI Assistant for a Quality Management System (SGC)** using a **production-oriented Retrieval-Augmented Generation (RAG)** architecture.

The assistant helps teachers consult official institutional procedures, grounding every answer strictly in real PDF documents while providing transparency, explainability, monitoring, and multilingual support through a Streamlit dashboard.

---

## 🎯 Use Case: AI Assistant for SGC Procedures  

The system supports professors who need to understand and correctly follow institutional procedures by:

- Answering questions **only from official SGC documents**
- Automatically retrieving relevant sections from PDF procedures
- Responding in the SAME language as the user (English or Spanish)
- Displaying consulted sources for transparency
- Showing similarity and groundedness metrics
- Monitoring system latency in real time
- Collecting user feedback on each response

The system works with **all procedures stored in the `SGC/` folder simultaneously (multi-procedure RAG).**

---

## 🌍 Multilingual Design  

1. User asks a question in English or Spanish.
2. The question is translated to Spanish for optimized retrieval.
3. Retrieval operates on official Spanish SGC documentation.
4. The final answer is generated in the SAME language as the original question.

This ensures:

- Accurate retrieval
- User-friendly interaction
- Strict contextual grounding

---

## 📁 Project Structure  

qms-rag-dashboard/
│
├── streamlit_app_template.py
├── requirements.txt
├── README.md
│
├── utils/
│ ├── rag.py
│ ├── qa_pipeline.py
│ └── explainability.py
│
└── SGC/
├── P-CA-05.pdf
├── P-CA-06.pdf
└── (other procedures...)


---

## 🧠 Core Technical Approach  

### 1️⃣ Document Ingestion (RAG)

- All PDFs inside `SGC/` are automatically loaded
- Text is extracted using **pypdf**
- Documents are split into chunks (~250 words)
- Each chunk is embedded using:

`all-MiniLM-L6-v2`

- A single in-memory vector knowledge base is built

---

### 2️⃣ Retrieval (utils/rag.py)

For each user question:

- The system retrieves the **Top-5 most relevant chunks**
- Cosine similarity is computed
- Threshold filtering removes weak matches
- Retrieved chunks are shown in the Explainability page

Metrics calculated:

- Max Similarity
- Average Similarity
- Groundedness (Top-3 similarity average)

---

### 3️⃣ Generation (utils/qa_pipeline.py)

- Detects user language
- Translates question for retrieval optimization
- Builds strict auditor prompt
- Injects retrieved context
- Forces response language to match original user input
- Calls LLM: `llama-3.1-8b-instant` (Groq API)
- Temperature set to `0.0` for deterministic output

---

### 4️⃣ Explainability (utils/explainability.py)

Includes:

- `explain_similarity()`
- `similarity_predict()`

Provides:

- Retrieved fragments display
- Similarity metrics
- Groundedness score
- Confidence estimation
- RAG influence visualization

Ensures transparency and audit-readiness.

---

### 5️⃣ Monitoring (Pillar III – Production Focus)

The Monitoring page tracks:

- Average Response Time (seconds)
- Total interactions

This aligns with production AI observability standards.

---

## 🧩 Trustworthy AI Principles in the System  

| Principle        | Implementation |
|------------------|---------------|
| Groundedness     | Answers restricted to retrieved chunks |
| Transparency     | Sources displayed to user |
| Explainability   | Similarity & groundedness metrics shown |
| Accountability   | Feedback mechanism included |
| Reliability      | Multi-document retrieval |
| Monitoring       | Average response time tracking |
| Human oversight  | Teachers can verify document sources |

---

## 🚀 How to Run the Project  

### 1️⃣ Clone Repository  

```bash
git clone https://github.com/danielhernandez-utt/qms-rag-dashboard/
cd qms-rag-dashboard
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Streamlit App
streamlit run app.py
Opens at:

http://localhost:8501
🌐 Deployment
Live App:
https://qms-rag-dashboard-w5xmdfrwxwxmqeyrnxhvyk.streamlit.app/

Repository:
https://github.com/danielhernandez-utt/qms-rag-dashboard/

⚠️ Limitations
Embeddings stored in memory (no persistent vector database)

Feedback not stored externally

Similarity explainability is descriptive (not SHAP/LIME)

Performance depends on PDF text quality

Translation step adds minor latency

🔮 Future Improvements
Persistent vector database (FAISS / Chroma)

Advanced monitoring dashboard

Feedback analytics storage

Role-based access

PDF version control

Response export to official templates

👥 Team
Daniel Alejandro Hernandez Castro — Streamlit Architecture & Monitoring

Emer Ignacio Bernal — Gradio Prototype

Iliana Marlen Meza Sánchez — Explainability Integration

Víctor Daniel Ortiz García — Deployment

© 2026 Trustworthy AI – Module 17 Project