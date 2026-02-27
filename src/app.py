import streamlit as st
import asyncio
from rag.main import AlzheimerRAG
import pandas as pd
import os

st.set_page_config(page_title="Alzheimer's RAG Agent", layout="wide")

@st.cache_resource
def get_rag_system():
    rag = AlzheimerRAG()
    return rag

def main():
    st.title("🧠 Alzheimer's Disease Research Assistant")
    st.sidebar.header("Settings")
    
    rag = get_rag_system()
    
    query = st.text_input("Enter your research question about Alzheimer's:", 
                          placeholder="e.g., What are the latest biomarkers for early detection?")
    
    if query:
        with st.spinner("Searching and generating answer..."):
            result = rag.query(query)
            
            st.markdown("### Answer")
            st.write(result["answer"])
            
            with st.expander("Show Sources"):
                for i, source in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:** {source.metadata.get('title', 'No Title')}")
                    st.write(f"Link: https://pubmed.ncbi.nlm.nih.gov/{source.metadata.get('pmid', '')}")
                    st.text(source.page_content[:300] + "...")
                    st.divider()

if __name__ == "__main__":
    main()
