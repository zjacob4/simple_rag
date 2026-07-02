from extraction import extract_full_text, extract_chunks, embed_chunks
import faiss
import numpy as np
import pickle

def add_knowledge():
    file = input("Provide the knowledge file address: ")
    full_text = extract_full_text(file)
    chunks = extract_chunks(full_text)
    embeddings, text_chunks = embed_chunks(chunks)
    
    # load index from store
    index = faiss.read_index("simple_rag.index")

    # get list of existing IDs
    stored_ids = faiss.vector_to_array(index.id_map)

    # get largest stored ID and start next at +1
    new_id = stored_ids[-1] + 1

    # pair IDs with text chunks
    text_lookup = {}
    embedding_lookup = {}
    for i in len(embeddings):
        text_lookup[i+new_id] = text_chunks[i]
        embedding_lookup[i+new_id] = embeddings[i]

    # create FAISS index
    f_index = faiss.IndexFlatL2(1536)
    f_index = faiss.IndexIDMap(f_index)
    f_index.add_with_ids(embeddings,embedding_lookup.keys())

    faiss.write_index(f_index, "simple_rag.index")
    with open("text_lookup.pkl", "wb") as f:
        pickle.dump(text_lookup, f)

    return f_index, text_lookup

if __name__ == "__main__":
    add_knowledge()