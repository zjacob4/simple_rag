from retreiver import pinecone_retrieval
import os
from dotenv import load_dotenv
from openai import OpenAI

# load env vars from .env
load_dotenv()
oai_key = os.getenv("OPENAI_KEY")

# initialize the client 
client = OpenAI(api_key=oai_key)

def search_with_context(query):

    retrieved_chunks = pinecone_retrieval(query)

    query_with_context = "The query is: "+query+" The retrieved context is: "+ ",".join(retrieved_chunks)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions="you are a RAG agent. You will be provided with a query and the RAG-retrieved context in the format of 'The query is' + query + 'The retrieved context is ' + the retrieved chunks. The retrieved chunks will include their RAG retrieval scores too in the form of 'text' with a score of 'score', from Pinecone. Use the query, text chunks, and text chunks scores to answer the initial query as best you can. However, do NOT use any information not found in the query + context to create your response.",
            input=query_with_context
        )
    except Exception as e:
        print(f"An error occurred: {e}")

    return response.output_text

if __name__ == "__main__":
    answer = search_with_context()
    print(answer)
