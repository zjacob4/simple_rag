import faiss

def create_new_index_file():
    # define vector dimension
    dimension = 1536

    # initialize a flat index (brute-force euclidean distance)
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap2(index)


    # save a blank index
    faiss.write_index(index, "simple_rag.index")

    print(f"Total vectors in file: {index.ntotal}")

    return index.ntotal

if __name__ == "__main__":
    create_new_index_file()