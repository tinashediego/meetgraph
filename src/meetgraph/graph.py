"""Neo4j knowledge graph: schema, ingestion, and graph queries.

Graph model:
  (Person)-[:ATTENDED]->(Meeting)-[:DISCUSSED]->(Topic)
  (Meeting)-[:DECIDED]->(Decision)
  (Meeting)-[:CREATED]->(ActionItem)-[:ASSIGNED_TO]->(Person)
  (Chunk {text, embedding})-[:PART_OF]->(Meeting)
"""

from dataclasses import asdict

from langchain_core.embeddings import Embeddings
from neo4j import Driver, GraphDatabase

from .config import settings
from .embeddings import DIMENSIONS
from .meetings import Meeting, chunk_meeting

VECTOR_INDEX = "chunk_embedding"

SCHEMA = [
    "CREATE CONSTRAINT meeting_id IF NOT EXISTS FOR (m:Meeting) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    f"""CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS FOR (c:Chunk) ON c.embedding
        OPTIONS {{indexConfig: {{`vector.dimensions`: {DIMENSIONS}, `vector.similarity_function`: 'cosine'}}}}""",
]

UPSERT_MEETING = """
MERGE (m:Meeting {id: $id})
SET m.title = $title, m.date = date($date)
WITH m
FOREACH (name IN $attendees |
  MERGE (p:Person {name: name})
  MERGE (p)-[:ATTENDED]->(m))
FOREACH (name IN $topics |
  MERGE (t:Topic {name: name})
  MERGE (m)-[:DISCUSSED]->(t))
FOREACH (i IN range(0, size($decisions) - 1) |
  MERGE (d:Decision {id: $id + '#decision-' + i})
  SET d.text = $decisions[i]
  MERGE (m)-[:DECIDED]->(d))
FOREACH (i IN range(0, size($action_items) - 1) |
  MERGE (a:ActionItem {id: $id + '#action-' + i})
  SET a.task = $action_items[i].task, a.due = date($action_items[i].due)
  MERGE (owner:Person {name: $action_items[i].owner})
  MERGE (m)-[:CREATED]->(a)
  MERGE (a)-[:ASSIGNED_TO]->(owner))
WITH m
UNWIND $chunks AS chunk
MERGE (c:Chunk {id: chunk.id})
SET c.text = chunk.text, c.seq = chunk.seq, c.embedding = chunk.embedding
MERGE (c)-[:PART_OF]->(m)
"""


def connect() -> Driver:
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        # Fail fast when Neo4j isn't running instead of retrying for minutes.
        connection_timeout=5,
        connection_acquisition_timeout=10,
    )
    driver.verify_connectivity()
    return driver


def ingest(driver: Driver, meetings: list[Meeting], embeddings: Embeddings) -> int:
    """Create the schema and upsert every meeting with its embedded chunks. Returns the chunk count."""
    for statement in SCHEMA:
        driver.execute_query(statement)

    total = 0
    for meeting in meetings:
        chunks = chunk_meeting(meeting)
        vectors = embeddings.embed_documents([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        record = asdict(meeting)
        record.pop("transcript")
        driver.execute_query(UPSERT_MEETING, **record, chunks=[asdict(c) for c in chunks])
        total += len(chunks)
    return total


def list_meetings(driver: Driver) -> list[dict]:
    records, _, _ = driver.execute_query(
        """
        MATCH (m:Meeting)
        OPTIONAL MATCH (p:Person)-[:ATTENDED]->(m)
        OPTIONAL MATCH (m)-[:DISCUSSED]->(t:Topic)
        RETURN m.id AS id, m.title AS title, toString(m.date) AS date,
               collect(DISTINCT p.name) AS attendees, collect(DISTINCT t.name) AS topics
        ORDER BY date
        """
    )
    return [r.data() for r in records]


def action_items(driver: Driver, person: str | None = None, topic: str | None = None) -> list[dict]:
    """Action items, optionally filtered by owner and/or a topic of the meeting that created them."""
    records, _, _ = driver.execute_query(
        """
        MATCH (m:Meeting)-[:CREATED]->(a:ActionItem)-[:ASSIGNED_TO]->(owner:Person)
        WHERE ($person IS NULL OR toLower(owner.name) CONTAINS toLower($person))
          AND ($topic IS NULL OR EXISTS {
                MATCH (m)-[:DISCUSSED]->(t:Topic) WHERE toLower(t.name) CONTAINS toLower($topic) })
        RETURN owner.name AS owner, a.task AS task, toString(a.due) AS due,
               m.title AS meeting, toString(m.date) AS meeting_date
        ORDER BY a.due
        """,
        person=person,
        topic=topic,
    )
    return [r.data() for r in records]


def topic_timeline(driver: Driver, topic: str) -> list[dict]:
    """Every meeting that discussed a topic, in date order, with what was decided there."""
    records, _, _ = driver.execute_query(
        """
        MATCH (m:Meeting)-[:DISCUSSED]->(t:Topic)
        WHERE toLower(t.name) CONTAINS toLower($topic)
        OPTIONAL MATCH (m)-[:DECIDED]->(d:Decision)
        RETURN m.title AS meeting, toString(m.date) AS date, collect(DISTINCT d.text) AS decisions
        ORDER BY date
        """,
        topic=topic,
    )
    return [r.data() for r in records]
