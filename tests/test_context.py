from langchain_core.documents import Document

from meetgraph.context import build_context


def hit(meeting_id, title, text, score):
    meta = {
        "meeting_id": meeting_id,
        "title": title,
        "date": "2026-09-10",
        "attendees": ["Priya Nair"],
        "decisions": ["Use SQLite"],
        "action_items": [{"owner": "Sam Okafor", "task": "Prototype storage", "due": "2026-09-24"}],
    }
    return Document(page_content=text, metadata=meta), score


def test_groups_by_meeting_best_score_first_and_dedupes():
    context = build_context([
        hit("a", "Roadmap", "Ava: offline first", 0.7),
        hit("b", "Design Review", "Sam: use SQLite", 0.9),
        hit("a", "Roadmap", "Ava: offline first", 0.7),
    ])
    assert context.index("## Design Review") < context.index("## Roadmap")
    assert context.count("Ava: offline first") == 1
    assert "- Sam Okafor: Prototype storage (due 2026-09-24)" in context
    assert "- Use SQLite" in context


def test_empty_hits():
    assert build_context([]) == "No matching meetings found."
