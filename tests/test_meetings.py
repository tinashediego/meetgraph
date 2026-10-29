from meetgraph.config import DEFAULT_DATA_DIR
from meetgraph.meetings import CHUNK_SIZE, chunk_meeting, load_meetings


def test_loads_sample_meetings():
    meetings = load_meetings(DEFAULT_DATA_DIR)
    assert len(meetings) == 4
    roadmap = next(m for m in meetings if m.id == "2026-09-02-q4-roadmap")
    assert "Priya Nair" in roadmap.attendees
    assert roadmap.action_items[0].owner == "Priya Nair"


def test_chunks_cover_transcript_with_stable_ids():
    meeting = load_meetings(DEFAULT_DATA_DIR)[0]
    chunks = chunk_meeting(meeting)
    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_SIZE for c in chunks)
    assert [c.id for c in chunks] == [f"{meeting.id}#{i}" for i in range(len(chunks))]
    assert [c.id for c in chunk_meeting(meeting)] == [c.id for c in chunks]
    # Every speaker turn ends up in some chunk.
    for line in meeting.transcript.splitlines():
        assert any(line[:40] in c.text for c in chunks)
