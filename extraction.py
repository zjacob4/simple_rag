from pypdf import PdfReader
from openai import OpenAI

client = OpenAI()

def extract_full_text(file):
    # extract pages from file
    pages = PdfReader(file)

    # compile text from each page into one large text
    full_text = ""
    for page in pages:
        full_text += page.extract_text()

    return full_text

def extract_chunks(full_text):
    
    chunks = []
    curr_chunk = ""
    for c in full_text:
        if c != ".":
            curr_chunk += c
        else:
            chunks.append(curr_chunk)
            curr_chunk = ""

    return chunks

# gets embeddings for one text snippet
def get_embeddings(text, model="text-embedding-3-small"):

    text = text.replace("\n","")
    return client.embeddings.create(input=text, model=model).data[0].embeddings

# runs get_embeddings iteratively across a list of chunks
def embed_chunks(chunks):

    vectorized_chunks = []
    for chunk in chunks:
        vectorized_chunks.append(get_embeddings(chunk))

    return vectorized_chunks



