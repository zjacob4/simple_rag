import streamlit as st

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
        with st.spinner("Thinking..."):
            answer = search_with_context(query)
        st.success(answer)

    # set session state for uploades
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()


    # upload file
    new_file = st.file_uploader("Choose a PDF", type="pdf")
    if new_file is not None:

        # identity for this upload
        file_id = (new_file.name, new_file.size)

        # check if file already indexed in this session
        if file_id in st.session_state.indexed_files:
            st.info(f"'{new_file.name}' is already indexed")
        elif st.button("Index this document"):
            with st.spinner("Indexing..."):
                len_vectors = add_knowledge(new_file)
            st.session_state.indexed_files.add(file_id)
        
            # show length of vectors added
            st.write(f"{len_vectors} vectors uploaded to vector store.")

    return 0

if __name__ == "__main__":
    homepage()