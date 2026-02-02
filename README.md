
# Module 15: Trustworthy AI Explainer Dashboard  
## Team Project – App Prototyping with Streamlit  

This project presents a **Trustworthy AI Assistant for a Quality Management System (SGC)** using **Retrieval-Augmented Generation (RAG)**.  

The assistant helps teachers consult official institutional procedures, grounding every answer in real PDF documents while providing transparency, explainability, and user feedback through both **Streamlit** and **Gradio** interfaces.

---

## 🎯 Use Case: AI Assistant for SGC Procedures  

The system supports professors who need to understand and correctly follow institutional procedures by:

- Answering questions **only from official SGC documents**  
- Automatically retrieving relevant sections from PDF procedures  
- Displaying consulted sources for transparency  
- Explaining how answers were generated  
- Collecting user feedback on each response  

The system works with **all procedures stored in the `SGC/` folder simultaneously (multi-procedure RAG).**

---

## 📁 Project Structure  

module-15/project/
├── module-15-project.md # Project specification
├── module-15-project-template.ipynb # Learning notebook with examples
├── gradio_app_template.py # Customized Gradio prototype (RAG-ready)
├── streamlit_app_template.py # Customized Streamlit dashboard (RAG-based)
├── requirements.txt # Python dependencies
├── utils/
│ └── rag.py # Multi-PDF RAG implementation
└── SGC/ # Institutional procedures (PDFs)
├── P-CA-05.pdf
├── P-CA-06.pdf
└── (other procedures...)


---

## 🧠 Core Technical Approach  

### 1️⃣ Document Ingestion (RAG)
- All PDFs inside `SGC/` are automatically loaded  
- Text is extracted with **pypdf**  
- Documents are split into chunks  
- Each chunk is embedded using: all-MiniLM-L6-v2


- A single in-memory vector knowledge base is built  

---

### 2️⃣ Retrieval  
For each user question:

- The system retrieves the **top-3 most relevant text chunks**  
- Chunks come from any procedure in the `SGC/` folder  
- Retrieved sources are shown in expandable panels in Streamlit  

---

### 3️⃣ Generation (LLM with Grounding)

- Retrieved context is injected into a structured system prompt  
- The model is restricted to answer **ONLY using retrieved procedure text**  
- Backend model: llama-3.1-8b-instant (Groq)


---

### 4️⃣ Explainability  

The system provides a transparency summary including:

- Input length  
- Output length  
- RAG rationale (how retrieval guided the answer)  
- Statement that no external knowledge was used  

---

### 5️⃣ Human-in-the-Loop Feedback  

Users can:

- Rate answers as 👍 Helpful or 👎 Not Helpful  
- Add optional comments  
- Feedback is stored in session state for analysis  

---

## 🚀 How to Run the Project  

### Activate virtual environment  

```bash
cd M15
venv310\Scripts\activate

Run Streamlit Dashboard
streamlit run streamlit_app_template.py


Opens at:

http://localhost:8501

Run Gradio Prototype
python gradio_app_template.py


Opens at:

http://localhost:7860

**🧩 Trustworthy AI Principles in the System**

Pinciple	Implementation
Groundedness	Answers restricted to retrieved chunks
Transparency	Sources displayed to user
Explainability	RAG rationale shown
Accountability	Feedback mechanism included
Reliability	Multi-document retrieval
Human oversight	Teacher can verify sources

**Limitations**

Embeddings are stored in memory (not a persistent database)

Feedback is not yet saved to an external system

Explainability is descriptive (not full SHAP/LIME visuals)

Performance depends on PDF text quality


**Future Improvements**

Possible extensions:

Persistent vector database (FAISS / Chroma)

True SHAP or LIME visualizations

Analytics dashboard for feedback

Procedure selector filter

Export answers to official templates

User authentication


Team Deliverables

Submit:

TeamName_Module15_Project.ipynb

TeamName_Module15_GradioApp.py

TeamName_Module15_StreamlitApp.py

requirements.txt

README.md ← this file

deployment_urls.txt (if deployed)


Resources

Gradio Docs

Streamlit Docs

Groq API

Sentence-Transformers
