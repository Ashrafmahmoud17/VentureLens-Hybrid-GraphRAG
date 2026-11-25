
import os
from pathlib import Path

import chromadb
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# ----------------------------------
# 1. Load environment variables
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# ----------------------------------
# 2. Configuration
# ----------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "skillbridge_embeddings"


# ----------------------------------
# 3. Connect to Neo4j
# ----------------------------------

driver = GraphDatabase.driver(
    NEO4J_URL,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

driver.verify_connectivity()

print("Connected to Neo4j!")


# ----------------------------------
# 4. Fetch all graph nodes
# ----------------------------------

QUERY = """
MATCH (n)
RETURN
    labels(n)[0] AS node_type,
    properties(n) AS props
ORDER BY node_type
"""

with driver.session(database=NEO4J_DATABASE) as session:
    result = session.run(QUERY)
    nodes = [record.data() for record in result]

print(f"Retrieved {len(nodes)} nodes from Neo4j")


# ----------------------------------
# 5. Prepare text documents
# ----------------------------------

documents = []
ids = []
metadatas = []

for node in nodes:

    node_type = node["node_type"]
    props = node["props"]

    if node_type == "Skill":

        node_id = props["skill_id"]

        text = (
            f"Skill: {props['skill_name']}. "
            f"Category: {props['category']}. "
            f"Description: {props['description']}"
        )

    elif node_type == "Course":

        node_id = props["course_id"]

        text = (
            f"Course: {props['course_name']}. "
            f"Difficulty: {props['difficulty']}. "
            f"Description: {props['description']}"
        )

    elif node_type == "Job":

        node_id = props["job_id"]

        text = (
            f"Job Role: {props['job_title']}. "
            f"Description: {props['description']}"
        )

    elif node_type == "Project":

        node_id = props["project_id"]

        text = (
            f"Project: {props['project_name']}. "
            f"Difficulty: {props['difficulty']}. "
            f"Description: {props['description']}"
        )

    else:
        continue

    documents.append(text)
    ids.append(node_id)

    metadatas.append({
        "node_type": node_type,
        "node_id": node_id
    })

print(f"Prepared {len(documents)} documents")


# ----------------------------------
# 6. Load embedding model
# ----------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded!")


# ----------------------------------
# 7. Generate embeddings
# ----------------------------------

embeddings = model.encode(
    documents,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embeddings generated!")
print("Embedding shape:", embeddings.shape)


# ----------------------------------
# 8. Store embeddings in ChromaDB
# ----------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Embeddings stored in ChromaDB!")


# ----------------------------------
# 9. Verify collection
# ----------------------------------

print("Collection count:", collection.count())

driver.close()

print("Step 3 completed successfully!")
