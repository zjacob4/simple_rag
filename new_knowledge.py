from extraction import extract_full_text, extract_chunks, embed_chunks, create_findex

def add_knowledge():
    file = input("Provide the knowledge file address: ")
    full_text = extract_full_text(file)
    chunks = extract_chunks(full_text)
    embeddings, text_lookup = embed_chunks(chunks)
    f_index = create_findex(embeddings)

    return f_index, text_lookup

if __name__ == "__main__":
    add_knowledge()