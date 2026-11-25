from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ----------------------------------
# 1. Configuration
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "skillbridge_embeddings"

MODEL_NAME = "all-MiniLM-L6-v2"


# ----------------------------------
# 2. Load Embedding Model
# ----------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded!")


# ----------------------------------
# 3. Connect to ChromaDB
# ----------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Connected to ChromaDB!")
print("Collection count:", collection.count())


# ----------------------------------
# 4. Define Search Function
# ----------------------------------

def search_similar_nodes(question, top_k=5):

    # Convert question into embedding
    query_embedding = model.encode(
        question,
        convert_to_numpy=True
    )

    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Display results
    print("\nQuestion:", question)
    print("\nTop Similar Results:")
    print("-" * 50)

    for i in range(len(results["ids"][0])):

        print(f"\nResult {i + 1}")

        print("ID:", results["ids"][0][i])

        print("Metadata:", results["metadatas"][0][i])

        print("Document:", results["documents"][0][i])

        print("Distance:", results["distances"][0][i])


# ----------------------------------
# 5. Test Search
# ----------------------------------

question = "I want to become a Machine Learning Engineer. What skills do I need?"

search_similar_nodes(question, top_k=5)