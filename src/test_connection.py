import os 
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URL = os.getenv("NEO4J_URL")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE")


driver = GraphDatabase.driver(URL, auth=(USERNAME, PASSWORD))


try:
    driver.verify_connectivity()
    print("Connection to Neo4j database successful!")

except Exception as e:
    print("Connection to Neo4j database failed:", e)

finally:
    driver.close()