
import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# =====================================
# 1. Configuration
# =====================================

load_dotenv()

URL = os.getenv("NEO4J_URL")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# =====================================
# 2. Load CSV Files
# =====================================

skills_df = pd.read_csv(DATA_DIR / "skills.csv")
courses_df = pd.read_csv(DATA_DIR / "courses.csv")
jobs_df = pd.read_csv(DATA_DIR / "jobs.csv")
prerequisites_df = pd.read_csv(DATA_DIR / "prerequisites.csv")
projects_df = pd.read_csv(DATA_DIR / "projects.csv")

print("All datasets loaded successfully!")

print("Skills:", len(skills_df))
print("Courses:", len(courses_df))
print("Jobs:", len(jobs_df))
print("Prerequisites:", len(prerequisites_df))
print("Projects:", len(projects_df))



# =====================================
# 3. Neo4j Connection
# =====================================

driver = GraphDatabase.driver(
    URL,
    auth=(USERNAME, PASSWORD)
)

driver.verify_connectivity()

print("Connected to Neo4j!")



# =====================================
# 4. Create Uniqueness Constraints
# =====================================

def create_constraints(session):

    queries = [
        """
        CREATE CONSTRAINT skill_id_unique IF NOT EXISTS
        FOR (s:Skill)
        REQUIRE s.skill_id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT course_id_unique IF NOT EXISTS
        FOR (c:Course)
        REQUIRE c.course_id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT job_id_unique IF NOT EXISTS
        FOR (j:Job)
        REQUIRE j.job_id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT project_id_unique IF NOT EXISTS
        FOR (p:Project)
        REQUIRE p.project_id IS UNIQUE
        """
    ]

    for query in queries:
        session.run(query).consume()

    print("Uniqueness constraints created!")


# =====================================
# 5. Create Skill Nodes
# =====================================


def create_skill_nodes(session):

    rows = skills_df.to_dict("records")

    query = """
    UNWIND $rows AS row

    MERGE (s:Skill {skill_id: row.skill_id})

    SET s.skill_name = row.skill_name,
        s.category = row.category,
        s.description = row.description
    """

    session.run(query, rows=rows).consume()

    print(f"Created/Updated {len(rows)} Skill Nodes")


# =====================================
# 6. Create Course Nodes
# =====================================

def create_course_nodes(session):

    rows = courses_df.to_dict("records")

    query = """
    UNWIND $rows AS row

    MERGE (c:Course {course_id: row.course_id})

    SET c.course_name = row.course_name,
        c.difficulty = row.difficulty,
        c.description = row.description
    """

    session.run(query, rows=rows).consume()

    print(f"Created/Updated {len(rows)} Course Nodes")


# =====================================
# 7. Create Job Nodes
# =====================================

def create_job_nodes(session):

    rows = jobs_df.to_dict("records")

    query = """
    UNWIND $rows AS row

    MERGE (j:Job {job_id: row.job_id})

    SET j.job_title = row.job_title,
        j.description = row.description
    """

    session.run(query, rows=rows).consume()

    print(f"Created/Updated {len(rows)} Job Nodes")


# =====================================
# 8. Create Project Nodes
# =====================================

def create_project_nodes(session):

    rows = projects_df.to_dict("records")

    query = """
    UNWIND $rows AS row

    MERGE (p:Project {project_id: row.project_id})

    SET p.project_name = row.project_name,
        p.difficulty = row.difficulty,
        p.description = row.description
    """

    session.run(query, rows=rows).consume()

    print(f"Created/Updated {len(rows)} Project Nodes")




# =====================================
# 9. Create Course-Skill Relationships
# =====================================

def create_course_skill_relationships(session):

    for _, row in courses_df.iterrows():

        skill_ids = str(row["skill_ids"]).split(";")

        query = """
        MATCH (c:Course {course_id: $course_id})
        MATCH (s:Skill {skill_id: $skill_id})
        MERGE (c)-[:TEACHES]->(s)
        """

        for skill_id in skill_ids:
            session.run(
                query,
                course_id=row["course_id"],
                skill_id=skill_id.strip()
            ).consume()

    print("Course-Skill relationships created!")


# =====================================
# 10. Create Job-Skill Relationships
# =====================================

def create_job_skill_relationships(session):

    for _, row in jobs_df.iterrows():

        skill_ids = str(row["required_skill_ids"]).split(";")

        query = """
        MATCH (j:Job {job_id: $job_id})
        MATCH (s:Skill {skill_id: $skill_id})
        MERGE (j)-[:REQUIRES]->(s)
        """

        for skill_id in skill_ids:
            session.run(
                query,
                job_id=row["job_id"],
                skill_id=skill_id.strip()
            ).consume()

    print("Job-Skill relationships created!")



# =====================================
# 11. Create Prerequisite Relationships
# =====================================

def create_prerequisite_relationships(session):

    query = """
    UNWIND $rows AS row

    MATCH (pre:Course {
        course_id: row.prerequisite_course_id
    })

    MATCH (target:Course {
        course_id: row.course_id
    })

    MERGE (pre)-[:PREREQUISITE_OF]->(target)
    """

    rows = prerequisites_df.to_dict("records")

    session.run(query, rows=rows).consume()

    print("Prerequisite relationships created!")



# =====================================
# 12. Create Project-Skill Relationships
# =====================================

def create_project_skill_relationships(session):

    for _, row in projects_df.iterrows():

        skill_ids = str(row["skill_ids"]).split(";")

        query = """
        MATCH (p:Project {project_id: $project_id})
        MATCH (s:Skill {skill_id: $skill_id})
        MERGE (p)-[:USES]->(s)
        """

        for skill_id in skill_ids:
            session.run(
                query,
                project_id=row["project_id"],
                skill_id=skill_id.strip()
            ).consume()

    print("Project-Skill relationships created!")



# =====================================
# 13. Execute Graph Construction
# =====================================

try:

    with driver.session(database=DATABASE) as session:

        # Create constraints
        create_constraints(session)

        # Create Nodes
        create_skill_nodes(session)
        create_course_nodes(session)
        create_job_nodes(session)
        create_project_nodes(session)

        print("\nAll Nodes created successfully!")

        # Create Relationships
        create_course_skill_relationships(session)
        create_job_skill_relationships(session)
        create_prerequisite_relationships(session)
        create_project_skill_relationships(session)

        print("\nAll Relationships created successfully!")

        # Verify Graph
        nodes = session.run(
            "MATCH (n) RETURN count(n) AS total"
        ).single()["total"]

        relationships = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS total"
        ).single()["total"]

        print("\n========== GRAPH SUMMARY ==========")
        print("Total Nodes:", nodes)
        print("Total Relationships:", relationships)

finally:
    driver.close()
