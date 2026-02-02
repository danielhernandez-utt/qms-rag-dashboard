# utils/rag.py
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------------------------------------------------
# LOAD EMBEDDING MODEL ONCE
# ------------------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------------------
# LOAD PDF TEXT
# ------------------------------------------------------------
def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# ------------------------------------------------------------
# CHUNKING FUNCTION
# ------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ------------------------------------------------------------
# BUILD MULTI-PDF KNOWLEDGE BASE
# ------------------------------------------------------------
def build_knowledge_base(pdf_folder: str):
    """
    Loads ALL PDFs inside `pdf_folder`, chunks them,
    computes embeddings, and keeps track of the source procedure.
    """

    all_chunks = []
    all_embeddings = []
    all_sources = []   # <-- stores the PDF name for each chunk

    # List all PDF files in folder
    pdf_files = [
        f for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    ]

    if len(pdf_files) == 0:
        raise ValueError(f"No PDF files found in folder: {pdf_folder}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)

        print(f"Loading: {pdf_file}")

        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text)

        embeddings = embedding_model.encode(chunks)

        # Store everything
        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)

        # Save source name for each chunk
        all_sources.extend([pdf_file] * len(chunks))

    return {
        "chunks": all_chunks,
        "embeddings": np.array(all_embeddings),
        "sources": all_sources
    }

# ------------------------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# ------------------------------------------------------------
def retrieve_relevant_chunks(question: str, knowledge_base: dict, top_k: int = 3):
    """
    Returns a list of (chunk, source_pdf) pairs.
    """

    q_emb = embedding_model.encode([question])

    sims = cosine_similarity(q_emb, knowledge_base["embeddings"])[0]

    top_indices = np.argsort(sims)[-top_k:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "chunk": knowledge_base["chunks"][i],
            "source": knowledge_base["sources"][i],
            "score": float(sims[i])
        })

    return results
