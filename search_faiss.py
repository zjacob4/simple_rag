from retreiver import faiss_retrieval
import os
from dotenv import load_dotenv
from openai import OpenAI

# load env vars from .env
load_dotenv()
oai_key = os.getenv("OPENAI_KEY")

# initialize the client 
client = OpenAI(api_key=oai_key)

def search_with_context():

    query = input("What is your query? ")
    retrieved_chunks = faiss_retrieval(query)

    query_with_context = "The query is: "+query+" The retrieved context is: "+ ".".join(retrieved_chunks)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions="you are a RAG agent. You will be provided with a query and the RAG-retrieved context in the format of 'The query is' + query + 'The retrieved context is ' + the retrieved chunks. Use this to answer the initial query as best you can. However, do NOT use any information not found in the query + context to create your response.",
            input=query_with_context
        )
    except Exception as e:
        print(f"An error occurred: {e}")

    return response.output_text

if __name__ == "__main__":
    answer = search_with_context()
    print(answer)
