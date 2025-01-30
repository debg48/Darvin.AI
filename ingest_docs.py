import os
import pdfplumber
import hashlib
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize model for creating embeddings
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Directory where PDFs are stored
pdf_folder = '/home/debg48/books'  # Replace with the folder path containing PDFs
index_file = 'faiss_index.index'
processed_files_file = 'processed_files.txt'

# Load or initialize the FAISS index
def load_faiss_index():
    print("Loading FAISS index...")
    if os.path.exists(index_file):
        print(f"Found existing FAISS index at {index_file}")
        index = faiss.read_index(index_file)
    else:
        print("No existing FAISS index found, creating a new one.")
        index = None
    return index

# Save FAISS index to disk
def save_faiss_index(index):
    print(f"Saving FAISS index to {index_file}")
    faiss.write_index(index, index_file)

# Check if a PDF file has already been processed
def check_if_processed(pdf_path):
    print(f"Checking if {pdf_path} has been processed...")
    if os.path.exists(processed_files_file):
        with open(processed_files_file, 'r') as f:
            processed_files = f.read().splitlines()
    else:
        processed_files = []
    
    # Return True if the PDF is already in the list of processed files
    processed = os.path.basename(pdf_path) in processed_files
    print(f"Processed status for {pdf_path}: {processed}")
    return processed

# Add a PDF to the processed list
def mark_as_processed(pdf_path):
    print(f"Marking {pdf_path} as processed.")
    with open(processed_files_file, 'a') as f:
        f.write(os.path.basename(pdf_path) + '\n')

# Extract text from a PDF file using pdfplumber
def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    print(f"Extracted text from {pdf_path}")
    return text

# Create embeddings from the extracted text
def create_embeddings(text):
    print("Creating embeddings for the extracted text...")
    # Split text into chunks (e.g., lines or paragraphs)
    text_chunks = text.split("\n")  # You can also split by paragraphs
    embeddings = model.encode(text_chunks)
    print(f"Created embeddings for {len(text_chunks)} chunks.")
    return embeddings, text_chunks

# Create or update the FAISS index with the new embeddings
def create_faiss_index(embeddings, existing_index=None):
    print("Creating or updating the FAISS index...")
    if existing_index is None:
        print("No existing index. Creating a new FAISS index.")
        dimension = embeddings.shape[1]  # Size of embeddings
        index = faiss.IndexFlatL2(dimension)
    else:
        print("Using the existing FAISS index.")
        index = existing_index

    # Add the new embeddings to the FAISS index
    index.add(embeddings)
    print(f"Added {embeddings.shape[0]} embeddings to the FAISS index.")
    return index

# Process all PDFs in the folder
def process_pdfs_in_folder():
    print(f"Processing PDFs from folder: {pdf_folder}")
    # Load existing FAISS index or create a new one
    index = load_faiss_index()

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            
            # Skip PDFs that have already been processed
            if check_if_processed(pdf_path):
                print(f"Skipping already processed PDF: {pdf_file}")
                continue

            # Extract text from the PDF
            text = extract_text_from_pdf(pdf_path)

            # Create embeddings for the extracted text
            embeddings, text_chunks = create_embeddings(text)
            embeddings = np.array(embeddings).astype('float32')

            # Create or update the FAISS index with new embeddings
            index = create_faiss_index(embeddings, existing_index=index)

            # Mark this PDF as processed
            mark_as_processed(pdf_path)
            print(f"Processed: {pdf_file}")

    # Save the updated FAISS index to disk
    save_faiss_index(index)

# Example usage
process_pdfs_in_folder()
