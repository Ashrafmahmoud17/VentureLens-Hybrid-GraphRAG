
import os
from pathlib import Path

from neo4j import GraphDatabase
from dotenv import load_dotenv


# ----------------------------------
# 1. Load Environment Variables
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

NEO4J_URI = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# ----------------------------------
# 2. Connect to Neo4j
# ----------------------------------

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

driver.verify_connectivity()

print("Connected to Neo4j!")


# ----------------------------------
# 3. Graph Retrieval Function
# ----------------------------------

def get_required_skills(job_id):

    query = """
    MATCH (j:Job {job_id: $job_id})-[:REQUIRES]->(s:Skill)

    RETURN
        j.job_title AS job_title,
        s.skill_id AS skill_id,
        s.skill_name AS skill_name

    ORDER BY s.skill_name
    """

    with driver.session(database=NEO4J_DATABASE) as session:

        result = session.run(
            query,
            job_id=job_id
        )

        skills = [record.data() for record in result]

    return skills


# ----------------------------------
# 4. Test Graph Retrieval
# ----------------------------------

job_id = "J03"

skills = get_required_skills(job_id)

print("\nRequired Skills:")

for skill in skills:
    print(
        f"{skill['skill_id']} - {skill['skill_name']}"
    )

print(f"\nTotal Skills: {len(skills)}")


# ----------------------------------
# 5. Close Connection
# ----------------------------------

driver.close()

print("\nGraph Retrieval completed!")
