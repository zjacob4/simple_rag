from extraction import extract_full_text, extract_chunks, embed_chunks
import faiss
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# load pinecone api key from .env
load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")

pc_client = Pinecone(api_key=pinecone_key)

def add_knowledge():

    # pinecone remote index
    index = pc_client.Index(host="https://simple-rag-b8g8ufl.svc.aped-4627-b74a.pinecone.io")

    # get file and embeddings
    file = input("Provide the knowledge file address: ")
    full_text = extract_full_text(file)
    chunks = extract_chunks(full_text)
    embeddings, text_chunks = embed_chunks(chunks)

    # get list/array of existing IDs, sorted
    stored_ids = []
    #stored_ids = index.list(namespace="__default__").sort()
    for page in index.list(namespace="__default__"):
        for id in page:
            stored_ids.append(int(id.id))
    stored_ids.sort()

    # get largest stored ID and start next at +1
    if len(stored_ids) > 0:
        new_id = stored_ids[-1] + 1
    else:
        new_id = 0

    # initialize vector list
    vectors_list = []

    # pair IDs with text chunks
    for i in range(len(embeddings)):
        full_vector = {}
        metadata_dict = {}
        metadata_dict["text"] = text_chunks[i]
        metadata_dict["source"] = file
        full_vector["id"] = str(i + new_id)
        full_vector["values"] = embeddings[i]
        full_vector["metadata"] = metadata_dict
        vectors_list.append(full_vector)

    # add new items to the opened index
    index.upsert(namespace="__default__",vectors=vectors_list)

    return len(vectors_list)

if __name__ == "__main__":
    add_knowledge()