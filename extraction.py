from pypdf import PdfReader

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


