#Install the necessary libraries and packages 
# (torch, transformers, sentence-tranformers, faiss-cpu, chromadb, langchain, pypdf, fastapi, uvicorn)
import os
import pypdf  #Module for extracting texts from PDFs

#Step 2: Load and Preprocess the PDF

def extract_text_from_pdf(pdf_path):
    """This function extracts texts from a given PDF file."""
    with open(pdf_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

#Testing the function
pdf_path = "/home/idika/rag_env/Oppenheimer-2006-Applied_Cognitive_Psychology.pdf"
if os.path.exists(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text[:1000])   #This prints the first 1_000 characters

else:
    print("PDF file not found!")



#Cleaning and Preprocessing the Extracted Text for Better Accuracy and Retrieval
import re

def clean_text(text):
    """
    This function cleans the text extracted from the PDF.
    - It removes extra spaces and new lines

    Argument:
    - Extracted text from the PDF

    Returns:
    Cleaned text
    """
    text = re.sub(r'\s+', ' ', text)  #Replaces multiple spaces/newlines with a single space
    text = text.strip()
    return text


#Testing the function
cleaned_text = clean_text(extracted_text)
print(cleaned_text[:1000])             #Prints the first 1_000 characters after cleaning


#Splitting the Text into Chunks for Better Retrieval
import textwrap
def chunk_text(text, max_chunk_size=512):
    """ 
    - This function splits the cleaned text into chunks based on paragraphs.
    - If a paragraph exceeds the max_chunk_size, it further splits it into smaller parts

    Args:
    - text (str): The text (cleaned text) to be split into chunks based on paragraphs
    - max_chunk_size (int): Maximum token size per chunk

    Returns:
    list: A list of paragraph-based text chunks
    """
    #split text into paragraphs usingdouble newline as the separator
    paragraphs = text.split("\n\n")   #Splitting by paragraph

    chunks = []   #Defining an empty list that will contain the chunks

    for para in paragraphs:
        para = para.strip()    #Removing empty strings and strip whitespace
        if para:
            if len(para) > max_chunk_size:
                #Further split long paragraphs while keeping sentence structure
                sub_chunks = textwrap.wrap(para, width=max_chunk_size)
                chunks.extend(sub_chunks)
            else:
                chunks.append(para)
    return chunks

#Testing the chunk_text function
text_chunks = chunk_text(cleaned_text)
print(f"Total Chunks: {len(text_chunks)}")
print(text_chunks[:2])     #Printing the first 2 chunks


#Step 3: Embed Text for Efficient Retrieval
#Importing the necessary libraries 
# (Note: I already installed these libraries to my venv)

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

#Loading a pre-trained Embedding Model (specifically all-MiniLM-L6-v2 from Hugging Face)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#Converting text chunks into Embeddings
#In other words, each text chunk will be transformed into a vector representation

def embed_text_chunks(text_chunks):
    """This function converts text chunks into embeddings."""
    return embedding_model.encode(text_chunks, convert_to_numpy=True)

#Testing the embed_text_chunks function
text_embeddings = embed_text_chunks(text_chunks)
print(f"Generated {text_embeddings.shape[0]} embeddings with {text_embeddings.shape[1]} dimensions.")

#Storing Embeddings for Fast Retrieval
#This will be accomplished by indexing the embeddings with FAISS

def create_faiss_index(embeddings):
    """This function creates a FAISS index for fast similarity search"""
    dim = embeddings.shape[1]          #Obtaining the embedding dimension
    index = faiss.IndexFlatL2(dim)     #This creates the FAISS index - an L2 (Euclidean) distance-based index to store the embeddings
    index.add(embeddings)              #This adds the embeddings to the index
    return index

#Testing the FAISS index creation function
faiss_index = create_faiss_index(text_embeddings)
print("FAISS index has been created successfully!")


#Step 4: Implementation of Retrieval and Querying
#Goal: To retrieve relevant text chunks when a user queries the system

#Encoding the user's query
#Note: Embedding the user's query enables its mathematical comparison with the stored document embeddings
def embed_query(query, model):
    """This function converts a user's query into an embedding."""
    return model.encode([query], convert_to_numpy=True)

#Testing the user's query encoding function (embed_query)
query = "What do most experts agree on when it comes to writing?"
query_embedding = embed_query(query, embedding_model)

#Retrieving similar chunks from FAISS
def retrieve_similar_chunks(query_embedding, faiss_index, text_chunks, top_k=3):
    """Function for retrieving the top_k most relevant text chunks from FAISS"""
    _, indices = faiss_index.search(query_embedding, top_k)    #Finding the closest embeddings
    return [text_chunks[i] for i in indices[0]]                #This returns the matching text chunks

#Retrieving the top 5 relevant chunks
retrieved_chunks = retrieve_similar_chunks(query_embedding, faiss_index, text_chunks, top_k=3)


#Printing the results
for i, chunk in enumerate(retrieved_chunks):
    print(f"Chunk {i+1}:\n{chunk}\n{'='*50}")

#Setep 5 - Generate Answers with an LLM (one of Hugging Face's LLM in this case)
#Importing the required modules from various libraries (Note: I have already installed the libraries)
from transformers import AutoModelForCausalLM, AutoTokenizer

#Defining the model's name to be used.
#Note: The chosen model is capable of answering questions based on provided context
model_name = "gpt2"

#Loading the tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully!")
except Exception as e:
    print("Error loading tokenizer:", e)

#Loading the Model 
model = AutoModelForCausalLM.from_pretrained(model_name)

#print(tokenizer)
print("GPT-2 Model and Tokenizer successfully loaded on CPU")

#Adding a padding token for the model ("GPT-2") since it does not have one by default
tokenizer.pad_token = tokenizer.eos_token

#Generating answers using the retrieved Chunks
#To do this, the query and retrieved text chunks will be formatted into a prompt for the model

def generate_answer(query, retrieved_chunks, tokenizer, model):
    """
    This function generates a response for the user using an LLM that relies on the retrieved chunk(s) of text as context

    Parameters:
    - query (str): this is the query received from the user (i.e. the user's question)
    - retrieved_chunks (list): these are the list of chunks of the text returned after comparing the embedded query with the data embeddings stored in the FAISS
                               these will be used as context to generate a prompt for the LLM
    - model: this is the initialized (loaded) generative model to be used
    - tokenizer: this is the defined model tokenizer

    Returns:
    - (str): the relevant response to the user
    """

    context = "\n\n".join(retrieved_chunks)  #Combines the retrieved text
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    #Tokenize the prompt, specifying padding and truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    #Ensuring we don't exceed token limits
    if input_ids.shape[1] >= 1024:
        print(f"Warning: Input token length ({input_ids.shape[1]}) exceeds the model limit. Truncating...")
        input_ids = input_ids[:, -1024:] #Keep only the last 2048 tokens
        attention_mask = attention_mask[:, -1024:]
    

    #Generating a response with attention mask for better reliability
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,    #Reduces repetitive words
        no_repeat_ngram_size=3,    #Prevents repeated phrases
        do_sample=True,            #Enable sampling
        temperature=0.7,           #Makes the output more deterministic
        top_k=50                   #Limits randomness
    )

    
    return tokenizer.decode(output[0], skip_special_tokens=True) if output is not None else "Error: Model failed to generate a response"
    
    

#Testing the generate_answer function
answer = generate_answer(query, retrieved_chunks, tokenizer, model)

print(f"Answer:\n{answer}")


#Step 6: Creating a Full RAG Pipeline
#This entails integrating the retrieval and response generation into a single function
#Hence for each query, the function will:
# - retriieve relevant document chunks
# - use the chunks alongside the query to develop a prompt for the LLM
# - generate an answer for the user

def rag_pipeline(query, embedding_model, faiss_index, text_chunks, retriever, tokenizer, model, generator, top_k):
    """
    This is a complete RAG pipeline that:
    1. Retrieves relevant document chunks for the query
    2. Generates an answer based on the retrieved context and query.

    Args:
    - query (str): this is the user's input question
    - embedding_model (str): initialized model to be used in embedding the user's query
    - faiss_index: a variable that stores the FAISS index of the data for fast similarity search
    - text_chunks: a variable that stores the cleaned text chunks of the data
    - retriever (function): this is an earlier defined function that retrieves relevan document chunks
    - tokenizer: tokenizer for the generative model (for generating the responses)
    - model: this is the chosen generative model (a pre-trained LLM model)
    - top_k (int): number of relevant chunks to be retrieved
    - generator (function): Function that uses our chosen generative LLM to generate a response based on the prompt(user's query and context)

    Returns:
    - str: Generated answer from the model
    """

    #Step 1: Embed the query using the same embedding model used in embedding the data
    embedded_query = embed_query(query, embedding_model)
    
    # Step 2: Retrieve relevant chunks
    retrieved_chunks = retriever(embedded_query, faiss_index, text_chunks, top_k)

    #Step 3: Use a generative model to generate the responses
    response = generator(query, retrieved_chunks, tokenizer, model) 

    #Step 4: Return the generated response
    return response

#Step 7: Evaluating the System's Responses
#In other words, accessing the quality, relevance, and accuracy of the answers generated by the RAG system.

#Testing the system with a variety of question types
sample_queries = [
    "What are the key takeaways from the document?",
    "Can you summarize the main points in a few sentences?",
    "What does the document say about what most experts agree on when it comes to writing",
    "Was any body mentioned in the document? If yes, what is their role?",
    "Are there any numerical statistics or figures in the document?"
]
for query in sample_queries:
    answer = rag_pipeline(query, embedding_model, faiss_index, text_chunks, retrieve_similar_chunks, tokenizer, model, generate_answer, top_k=3)
    print(f"**Query:** \n{query}\n\n**Generated Answer:** \n\n{answer}\n{'='*50}\n\n")

