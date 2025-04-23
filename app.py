import os
import openai
import PyPDF2
import faiss
import streamlit as st
import numpy as np

# Set your OpenAI key (or use env var)
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Neurology Book Q&A")
st.title("📘 Ask Questions About the Neurology Book")

# ------------------- Load and Process PDF -------------------

@st.cache_resource
def load_book(pdf_path):
    reader = PyPDF2.PdfReader(open(pdf_path, "rb"))
    raw_chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        lines = text.split("\n")
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
    def get_openai_embedding(text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return np.array(response["data"][0]["embedding"])

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

    embeddings = np.array([get_openai_embedding(c) for c in chunks])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return chunks, metadata, index

# ------------------- Search + GPT Answer -------------------

def get_context(question, top_k=5, max_tokens=3000):
    q_embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")["data"][0]["embedding"]
    D, I = index.search(np.array([q_embedding]), k=top_k)

    context_parts = []
    references = []
    total_tokens = 0

    for i in I[0]:
        chunk = chunks[i]
        chunk_tokens = len(chunk.split())
        if total_tokens + chunk_tokens > max_tokens:
            break
        context_parts.append(chunk)
        meta = metadata[i]
        if meta:
            start = meta[0]
            end = meta[-1]
            references.append(f"Page {start['page']}, Line {start['line']}-{end['line']}")
        total_tokens += chunk_tokens

    return "\n\n".join(context_parts), references

def ask_openai(context, question):
    messages = [
        {
            "role": "system",
            "content": "You are an expert medical assistant helping a neurology student understand a textbook."
        },
        {
            "role": "user",
            "content": f"""
Based on the provided context, answer the question in a clear, detailed, and structured way.

- If possible, break down the answer into **bullet points**.
- Be concise, but informative.
- Only use information supported by the context.
- Do not add external information.
- If the question cannot be answered from the context, say: "The answer is not available in the provided context."

Context:
{context}

Question:
{question}

Answer in detailed points:
"""
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_tokens=700
    )
    return response.choices[0].message["content"].strip()

# ------------------- Run Streamlit -------------------

if not os.path.exists("book.pdf"):
    st.error("Please upload your book as book.pdf.")
else:
    raw_chunks = load_book("book.pdf")
    chunks, metadata, index = embed_chunks(raw_chunks)

    question = st.text_input("💬 Ask a question:")
    if st.button("Ask") and question:
        with st.spinner("Finding answer..."):
            context, refs = get_context(question)
            answer = ask_openai(context, question)
            st.success(answer)
            if refs:
                st.info("📄 Sources: " + "; ".join(refs))
