import streamlit as st
import requests

st.set_page_config(page_title="Neurology Book Q&A")

st.title("🧠 Ask Your Neurology Book")

question = st.text_input("Ask a question based on the book:")

if st.button("Ask"):
    if not question:
        st.warning("Please type a question.")
    else:
        with st.spinner("Thinking..."):
            res = requests.post("http://localhost:8000/ask", json={"question": question})
            if res.status_code == 200:
                data = res.json()
                st.success(data["answer"])
                st.info(f"📄 **Sources:** {data['sources']}")
            else:
                st.error("Something went wrong.")

