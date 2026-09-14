
import streamlit as st

st.set_page_config(page_title="SAP MM RAG Assistant")

st.title("SAP MM RAG Assistant 🤖")
st.write("Ask your SAP MM questions below.")

question = st.text_input("Enter your SAP MM question:")

if question:
    st.write("You asked:", question)
    st.info("RAG answer module will be connected in the next step.")