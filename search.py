from extraction import extract_full_text, extract_chunks, embed_chunks

def add_knowledge():
    file = input("Provide the knowledge file address: ")
    full_text = extract_full_text(file)
    chunks = extract_chunks(full_text)
    embeddings = embed_chunks(chunks)

