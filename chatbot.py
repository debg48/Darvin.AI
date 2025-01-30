import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, logging
import os

# Suppress warnings from transformers library
logging.set_verbosity_error()

# Initialize the SentenceTransformer model for embeddings
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Load a smaller, more efficient conversational model (EleutherAI GPT-Neo 1.3B)
chat_model = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', 
                      truncation=True, 
                      pad_token_id=50256)  # Set pad_token_id

# Check if the FAISS index exists
index_file = 'faiss_index.index'
if os.path.exists(index_file):
    index = faiss.read_index(index_file)
else:
    print("FAISS index not found. Please process the documents first.")
    index = None  # If index doesn't exist, it will be handled in the retrieval

# Function to retrieve relevant documents based on the query
def retrieve_relevant_documents(query, k=5):
    if index is None:
        print("No FAISS index available. Please process the documents first.")
        return []

    # Create an embedding for the query
    query_embedding = model.encode([query])

    # Search the FAISS index for similar documents
    D, I = index.search(np.array(query_embedding).astype('float32'), k)

    # Fetch the corresponding document chunks (For simplicity, we'll use a dummy chunk of text here)
    relevant_docs = [f"Document {i}: {I[0][i]}" for i in range(len(I[0]))]
    return relevant_docs

# Function to generate an answer using the smaller, lightweight model (GPT-Neo 1.3B)
def generate_answer(query, relevant_docs=None):
    if relevant_docs:
        # If relevant documents exist, use RAG-based generation
        context = "\n".join(relevant_docs)
        prompt = f"Answer this question based on the context: {query}\nContext: {context}"
    else:
        # If no relevant documents, proceed as a normal chatbot query
        prompt = query  # Directly use the query for response

    # Generate a response using the chat model
    response = chat_model(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return response

# Function to start a conversation
def chatbot():
    print("Welcome to the Chatbot! Ask anything, and I'll help you find the answer.")
    print("Type 'exit' to end the conversation.")
    print("If you want me to look into your books, ask something like 'look into my books'. Otherwise, I can chat with you!")

    while True:
        # Get user input (query)
        query = input("\nYou: ")

        # Exit condition
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        # Check if the query asks to search PDFs
        if any(phrase in query.lower() for phrase in ["look into my books", "search my books", "search in my pdfs"]):
            print("Bot: Searching through your books...")

            # Retrieve relevant documents for the query
            relevant_docs = retrieve_relevant_documents(query)

            # If no relevant documents found, continue to the next iteration
            if not relevant_docs:
                print("Bot: Sorry, I couldn't find any relevant documents.")
                continue

            # Generate an answer based on the query and relevant documents
            answer = generate_answer(query, relevant_docs)

            # Output the answer
            print("Bot:", answer)

        else:
            # For regular chatbot queries, generate a direct response
            answer = generate_answer(query)

            # Output the answer
            print("Bot:", answer)

# Start the chatbot
if __name__ == "__main__":
    chatbot()