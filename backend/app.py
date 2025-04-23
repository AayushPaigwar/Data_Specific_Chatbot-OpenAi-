import openai
import faiss
import PyPDF2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

# Allow Streamlit frontend to access FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

# Load and process the PDF (we do this only once at startup)
pdf_path = "book.pdf"
model = SentenceTransformer('all-MiniLM-L6-v2')

# Stores all extracted text and source references
raw_chunks = []

def load_pdf(path):
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    raw_chunks.append({
                        "text": line.strip(),
                        "page": page_num + 1,
                        "line": i + 1
                    })

def chunk_text(max_tokens=80):
    # Combine nearby lines into a chunk until hitting token threshold
    chunks = []
    metadata = []

    buffer = ""
    meta_buffer = []
    tokens = 0

    for entry in raw_chunks:
        line = entry["text"]
        line_tokens = len(line.split())

        if tokens + line_tokens > max_tokens:
            if buffer:
                chunks.append(buffer.strip())
                metadata.append(meta_buffer)
            buffer = line
            meta_buffer = [entry]
            tokens = line_tokens
        else:
            buffer += " " + line
            meta_buffer.append(entry)
            tokens += line_tokens

    if buffer:
        chunks.append(buffer.strip())
        metadata.append(meta_buffer)

    return chunks, metadata


# book_text = load_pdf(pdf_path)
load_pdf(pdf_path)
chunks, chunk_metadata = chunk_text()
embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

@app.post("/ask")
def ask_question(req: QuestionRequest):
    q_embedding = model.encode([req.question])
    D, I = index.search(np.array(q_embedding), k=10)

    def num_tokens(text):
        return len(text.split())

    context_chunks = []
    references = []
    total_tokens = 0

    for i in I[0]:
        chunk = chunks[i]
        chunk_tokens = num_tokens(chunk)
        if total_tokens + chunk_tokens > 3000:
            break
        context_chunks.append(chunk)
        
        # Get metadata for that chunk
        refs = chunk_metadata[i]
        if refs:
            start = refs[0]
            end = refs[-1]
            ref_text = f"Page {start['page']}, Line {start['line']}-{end['line']}"
            references.append(ref_text)
        
        total_tokens += chunk_tokens

    context = "\n\n".join(context_chunks)
    sources = "; ".join(references)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to a neurology textbook."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
        ],
        temperature=0.5,
        max_tokens=500
    )

    answer = response.choices[0].message["content"].strip()

    return {
        "answer": answer,
        "sources": sources
    }
