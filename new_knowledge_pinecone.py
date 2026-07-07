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
    index = pc_client.Index(host="INDEX_HOST")

    # get file and embeddings
    file = input("Provide the knowledge file address: ")
    full_text = extract_full_text(file)
    chunks = extract_chunks(full_text)
    embeddings, text_chunks = embed_chunks(chunks)

    # get list/array of existing IDs, sorted
    stored_ids = index.list(namespace="__default__").sort()

    # get largest stored ID and start next at +1
    if stored_ids.size > 0:
        new_id = stored_ids[-1] + 1
    else:
        new_id = 0

    # initialize underlying text chunks
    text_lookup = {}

    # pair IDs with text chunks
    for i in range(len(embeddings)):
        full_vector = {}
        metadata_dict = {}
        metadata_dict["text"] = text_chunks[i]
        metadata_dict["source"] = file
        full_vector["id"] = i + new_id
        full_vector["values"] = embeddings[i]
        full_vector["metadata"] = metadata_dict

    # add new items to the opened index
    

    return index, text_lookup

if __name__ == "__main__":
    add_knowledge()