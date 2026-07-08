import streamlit as st
import pandas as pd

from search_pinecone import search_with_context
from new_knowledge_pinecone import add_knowledge

def homepage():

    # page text
    st.title("Simple Rag")
    st.write("Welcome to the homepage")

    # store user query
    query = st.text_input("Query",label_visibility="collapsed",placeholder="Ask away...")

    # if user hits submit
    if st.button("Submit"):
        answer = search_with_context(query)
        st.success(answer)

    # upload file
    new_file = st.file_uploader("Choose a PDF", type="pdf")
    if new_file is not None:
        len_vectors = add_knowledge(new_file)
        
        # show length of vectors added
        st.write(f"{len_vectors} vectors uploaded to vector store.")

    return 0

if __name__ == "__main__":
    homepage()