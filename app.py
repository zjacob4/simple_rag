import streamlit as st
import pandas as pd

from search_pinecone import search_with_context

def homepage():

    # page text
    st.title("Simple Rag")
    st.write("Welcome to the homepage")

    # store user query
    query = st.text_input("",placeholder="Ask away...")

    # if user hits submit
    if st.button("Submit"):
        answer = search_with_context(query)
        st.success(answer)

    return 0

if __name__ == "__main__":
    homepage()