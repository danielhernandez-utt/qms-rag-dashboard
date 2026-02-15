# utils/rag.py
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model once for global efficiency
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding_model():
    return embedding_model

def load_pdf_text(pdf_path: str) -> str:
    # Extracts text from PDF using pypdf
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50):
    # Splits text into chunks with overlap to avoid losing context between fragments
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_knowledge_base(pdf_folder: str):
    # Initializes and populates the local vector store from PDF files
    all_chunks = []
    all_embeddings = []
    all_sources = []

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        return {"chunks": [], "embeddings": np.array([]), "sources": []}

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text)
        
        if chunks:
            # Generate embeddings for each chunk
            embeddings = embedding_model.encode(chunks)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
            all_sources.extend([pdf_file] * len(chunks))

    return {
        "chunks": all_chunks,
        "embeddings": np.array(all_embeddings),
        "sources": all_sources
    }



def retrieve_relevant_chunks(question: str, knowledge_base: dict, top_k: int = 5):
    if not knowledge_base["chunks"]:
        return []

    q_emb = embedding_model.encode([question])
    sims = cosine_similarity(q_emb, knowledge_base["embeddings"])[0]
    #print("DEBUG max similarity:", max(sims))

    # NEW: Adjust threshold to allow cross-language matching (0.35 is safer for EN-ES)
    threshold = 0.35 
    
    top_indices = np.argsort(sims)[-top_k:][::-1]
    results = []
    for i in top_indices:
        if sims[i] > threshold:
            results.append({
                "chunk": knowledge_base["chunks"][i],
                "source": knowledge_base["sources"][i],
                "score": float(sims[i])
            })
    return results