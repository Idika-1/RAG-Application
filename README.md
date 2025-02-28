# RAG-Application
Retrieval Augmented Generation (RAG) Application.

In this mini-project I developed a Retrieval Augmented Generation (RAG) system that leverages:  
- a transformer model to embed data and user’s query;
- a Large Language Model (LLM) to curate response to a user’s query based on the prompt (user’s query and context from the data) fed to the LLM.

The steps I followed are: 
**Step 1: Environment Setup**  
-	I ensured that I had Python installed on my Virtual Machine.   
-	I used a VM running on an Ubuntu OS on an Oracle VirtualBox.  
-	Setup and activated a virtual environment  
-	Installed all required libraries  
-	Configured VS Code for the project

**Step 2: Load and Preprocess PDF**  
-	Extracted text from a sample PDF using the pypdf library.  
-	Cleaned and preprocessed the extracted text.  
-	Chunked the text into paragraphs for better retrieval.

**Step 3: Embedded Test for Retrieval**  
-	Embedding’s are numerical representations of text in a high-dimensional space.  
-	I utilized Hugging Face’s “all-MiniLM-L6-v2” SentenceTransformer model for the embedding.

**Step 4: Retrieval and Querying Implementation**  
-	Used FAISS index to index the embedded text chunks for faster retrieval.  
-	Used the same SentenceTransformer model to embed the user’s query.  
-	Used the FAISS index to perform a similarity search of the embedded user’s query on the FAISS indexed embedded text chunks from the PDF data.  
-	Returned the most relevant text chunks (top_k= 3) to the user’s query.

**Step 5: Answer Generation with an LLM**  
-	Combined user’s query and retrieved text chunks to create a prompt for an LLM to generate answers.  
-	Utilized a light-weight open source LLM to generate responses to user’s queries based on the supplied prompt.

**Step 6: Created a Full RAG Pipeline**  
-	This involved integrating the retrieval and response generation into a single function.  


