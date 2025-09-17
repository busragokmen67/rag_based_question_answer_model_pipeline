# rag_based_question_answer_model_pipeline

📌 Features

🗂 Load and clean .txt documents

📏 Auto-chunk based on the smallest document

🧠 Embed using sentence-transformers (MiniLM)

📈 Retrieve top-k most relevant chunks via cosine similarity

💬 Answer questions using falcon-rw-1b (local, CPU-friendly)





📁 Directory Structure

Place your .txt files inside a folder. Example:

project-root/
├── qa.py
├── docs/
│   ├── api.txt
│   ├── usage.txt

directory = '/path/to/docs'



⚙️ Installation


Make sure you are using Python 3.8+. Then install the required packages:


!pip install sentence-transformers scikit-learn transformers torch


I used Google Colab, you may want to restart the runtime after installing.




🚀 How It Works


1. Load and Preprocess
   
texts = load_and_preprocess_text_files(directory)

This reads all .txt files, removes punctuation and whitespace, and stores them as a list.

2. Chunking
   
chunks = split_text_into_chunks_based_on_min_file(texts)

The system finds the smallest file and uses its word count to split all other files into equally sized chunks.

Minimum Words per Chunk: 18 

Total Chunks: 22 


Examples:


Chunk 1: The HTTP API supports three endpoints POST apiindex build or rebuild the index POST apiquery payload question top_k... 

Chunk 2: GET apistatus returns ready true when the index is ready Before running set your API keys as environment... 

Chunk 3: variables Use OPENAI_API_KEY for OpenAI models and LLAMA_API_KEY for hosted Llama endpoints To use a local embedding model... 

3. Embedding with Sentence-BERT
   
e_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = e_model.encode(chunks, convert_to_tensor=True)

Generates sentence embeddings for each chunk.

4. Query + Retrieval
   
top_chunks, _ = retrieve_top_k_chunks_cosine(query, e_model, embeddings, chunks, k=2)

Finds the top k most similar chunks using cosine similarity between your query and the document embeddings.


5. Generate Answer with LLM
   
answer = generate_answer_directly(query, top_chunks)

I used a local language model (falcon-rw-1b that is a type of small langchain model) to generate a contextual answer from the top chunks.


❓ Example

Given these input .txt files:

Chunk 16: To reset your password open Settings and choose Security Click Forgot Password and enter your registered email address

Chunk 17: Check your inbox for the reset link and follow the instructions If no email arrives within five minutes


query = "How do I reset my password?"

Most relevant answer: To reset your password open Settings and choose Security Click Forgot Password and enter your registered email address  

Second relevant answer: Check your inbox for the reset link and follow the instructions If no email arrives within five minutes  

🔍 Question: How do I reset my password?  

🧠 Answer(with LLM): Changing your password will ensure you are the only person who can access your account, and will help improve your security..

⚠️ Notes;

The default model falcon-rw-1b is CPU-friendly, but slow. This is a form of small LangChain.


📦 Requirements;

sentence-transformers	      ≥2.2,
transformers	              ≥4.30,
torch	                      ≥1.12,
scikit-learn	              ≥1.0



🧠 Attribution;

SentenceTransformers

Hugging Face Transformers

Falcon-RW-1B Model
