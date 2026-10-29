"""Turns GraphRAG search hits into a compact, citable context block for the LLM (or an MCP client)."""

from langchain_core.documents import Document


def build_context(hits: list[tuple[Document, float]]) -> str:
    """Group hits by meeting (best match first) and render each meeting's graph facts plus its excerpts."""
    meetings: dict[str, dict] = {}
    for doc, score in hits:
        meta = doc.metadata
        entry = meetings.setdefault(meta["meeting_id"], {"meta": meta, "best": score, "excerpts": []})
        entry["best"] = max(entry["best"], score)
        if doc.page_content not in entry["excerpts"]:
            entry["excerpts"].append(doc.page_content)

    if not meetings:
        return "No matching meetings found."

    blocks = []
    for entry in sorted(meetings.values(), key=lambda e: e["best"], reverse=True):
        meta = entry["meta"]
        lines = [
            f"## {meta['title']} ({meta['date']})",
            f"Attendees: {', '.join(meta['attendees']) or 'unknown'}",
        ]
        if meta["decisions"]:
            lines.append("Decisions:")
            lines += [f"- {d}" for d in meta["decisions"]]
        if meta["action_items"]:
            lines.append("Action items:")
            lines += [f"- {a['owner']}: {a['task']} (due {a['due']})" for a in meta["action_items"]]
        lines.append("Transcript excerpts:")
        lines += [f"> {line}" for excerpt in entry["excerpts"] for line in excerpt.splitlines()]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
