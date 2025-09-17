import argparse
import os
import re
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1. Load and Preprocess Text Files ==========
def preprocess_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    return text

def load_and_preprocess_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                raw_text = file.read()
                cleaned_text = preprocess_text(raw_text)
                texts.append(cleaned_text)
    return texts

# ========== 2. Chunk Text ==========
def calculate_min_file_word_count(texts):
    return min(len(text.split()) for text in texts)

def split_text_into_chunks_based_on_min_file(texts):
    min_words_per_chunk = calculate_min_file_word_count(texts)
    chunks, current_chunk = [], []
    for text in texts:
        words = text.split()
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= min_words_per_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))  # add any remaining words
    return chunks

# ========== 3. Embedding ==========
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return model, embeddings

# ========== 4. Retrieval ==========
def retrieve_top_k_chunks_cosine(query, embed_model, embeddings, chunks, k=3):
    query_embedding = embed_model.encode([query], convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k_indices]

# ========== 5. Generation ==========
def load_generator_model():
    model_name = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )
    return generator

def generate_answer(query, top_chunks, generator):
    context = "\n\n".join(top_chunks)
    prompt = f"""Use the following context to answer the question:

Context:
{context}

Question: {query}
Answer:"""
    result = generator(prompt)[0]["generated_text"]
    return result.split("Answer:")[-1].strip()

# ========== 6. Main CLI Function ==========
def main():
    parser = argparse.ArgumentParser(description="RAG CLI Q&A")
    parser.add_argument('--question', type=str, required=True, help='Your question for the system')
    parser.add_argument('--docs_path', type=str, default='./docs', help='Path to directory containing text files')
    args = parser.parse_args()

    print("🔄 Loading and preprocessing documents...")
    texts = load_and_preprocess_text_files(args.docs_path)

    print("🧩 Splitting into chunks...")
    chunks = split_text_into_chunks_based_on_min_file(texts)

    print("📐 Embedding chunks...")
    embed_model, embeddings = embed_chunks(chunks)

    print("🔍 Retrieving top chunks...")
    top_chunks = retrieve_top_k_chunks_cosine(args.question, embed_model, embeddings, chunks, k=3)

    print("🧠 Generating answer...")
    generator = load_generator_model()
    answer = generate_answer(args.question, top_chunks, generator)

    print("\n💬 Answer:", answer)

if __name__ == '__main__':
    main()







