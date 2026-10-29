"""End-to-end against a running Neo4j (docker compose up -d). Skipped when Neo4j isn't reachable."""

import pytest

from meetgraph import graph, retriever
from meetgraph.config import DEFAULT_DATA_DIR
from meetgraph.context import build_context
from meetgraph.embeddings import LocalEmbeddings
from meetgraph.meetings import load_meetings


@pytest.fixture(scope="module")
def driver():
    try:
        d = graph.connect()
    except Exception as exc:  # noqa: BLE001 - any connection failure means "no Neo4j here"
        pytest.skip(f"Neo4j not reachable: {exc}")
    graph.ingest(d, load_meetings(DEFAULT_DATA_DIR), LocalEmbeddings())
    yield d
    d.close()


def test_semantic_search_expands_to_graph_facts(driver):
    store = retriever.build_store(LocalEmbeddings())
    hits = retriever.search(store, "How will we resolve conflicting edits from two devices?", k=3)
    assert "2026-09-10-offline-design-review" in {doc.metadata["meeting_id"] for doc, _ in hits}
    context = build_context(hits)
    assert "Use last-writer-wins per field with a server-side audit log" in context  # a Decision node, not transcript text


def test_action_items_by_person_and_topic(driver):
    items = graph.action_items(driver, person="marcus", topic="offline")
    assert {i["task"] for i in items} == {
        "Estimate backend sync API changes for offline mode",
        "Add a version field to every record in the sync API",
    }


def test_topic_timeline_is_ordered(driver):
    timeline = graph.topic_timeline(driver, "offline mode")
    assert [t["date"] for t in timeline] == ["2026-09-02", "2026-09-10", "2026-10-06"]
