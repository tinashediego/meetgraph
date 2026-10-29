"""GraphRAG retrieval: vector search finds transcript chunks, then a graph traversal pulls in
each chunk's meeting, attendees, decisions and action items."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jVector

from .config import settings
from .graph import VECTOR_INDEX

# Runs once per matched chunk; LangChain binds `node` (the Chunk) and `score`.
GRAPH_EXPANSION = """
MATCH (node)-[:PART_OF]->(m:Meeting)
OPTIONAL MATCH (p:Person)-[:ATTENDED]->(m)
WITH node, score, m, collect(DISTINCT p.name) AS attendees
OPTIONAL MATCH (m)-[:DECIDED]->(d:Decision)
WITH node, score, m, attendees, collect(DISTINCT d.text) AS decisions
OPTIONAL MATCH (m)-[:CREATED]->(a:ActionItem)-[:ASSIGNED_TO]->(o:Person)
WITH node, score, m, attendees, decisions,
     [x IN collect(DISTINCT {owner: o.name, task: a.task, due: toString(a.due)}) WHERE x.task IS NOT NULL] AS actions
RETURN node.text AS text, score,
       {chunk_id: node.id, meeting_id: m.id, title: m.title, date: toString(m.date),
        attendees: attendees, decisions: decisions, action_items: actions} AS metadata
"""


def build_store(embeddings: Embeddings) -> Neo4jVector:
    return Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password,
        index_name=VECTOR_INDEX,
        retrieval_query=GRAPH_EXPANSION,
    )


def search(store: Neo4jVector, query: str, k: int = 5) -> list[tuple[Document, float]]:
    return store.similarity_search_with_score(query, k=k)
