import os
import PyPDF2
import faiss
import openai
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

# Set your OpenAI key from environment or directly here
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Neurology Book Q&A")
st.title("🧠 Ask Anything About Your Neurology Book")

# ------------------ PDF & Embedding Setup ------------------

@st.cache_resource
def load_book(path):
    reader = PyPDF2.PdfReader(open(path, "rb"))
    raw_chunks = []
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
    return raw_chunks

@st.cache_resource
def embed_chunks(raw_chunks, max_tokens=80):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks, metadata = [], []
    buffer, meta_buffer, tokens = "", [], 0

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

    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))

    return chunks, metadata, index, model

# Load PDF and process
raw_chunks = load_book("book.pdf")
chunks, chunk_metadata, faiss_index, embed_model = embed_chunks(raw_chunks)

# ------------------ Question Answering ------------------

def get_relevant_context(question, top_k=5, max_context_tokens=3000):
    q_vector = embed_model.encode([question])
    D, I = faiss_index.search(np.array(q_vector), k=top_k)

    selected_chunks = []
    references = []
    total_tokens = 0

    for i in I[0]:
        chunk = chunks[i]
        meta = chunk_metadata[i]
        chunk_tokens = len(chunk.split())
        if total_tokens + chunk_tokens > max_context_tokens:
            break
        selected_chunks.append(chunk)
        if meta:
            start = meta[0]
            end = meta[-1]
            ref = f"Page {start['page']}, Line {start['line']}-{end['line']}"
            references.append(ref)
        total_tokens += chunk_tokens

    return "\n\n".join(selected_chunks), references

def ask_openai(context, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to a neurology textbook."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0.5,
    )
    return response.choices[0].message["content"].strip()

# ------------------ Streamlit UI ------------------

question = st.text_input("💬 Ask a question about the book:")
if st.button("Ask") and question:
    with st.spinner("Searching and answering..."):
        context, refs = get_relevant_context(question)
        answer = ask_openai(context, question)

        st.success(answer)
        if refs:
            st.info("📄 References: " + "; ".join(refs))
