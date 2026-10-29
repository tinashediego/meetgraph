import asyncio

from meetgraph.mcp_server import mcp


def test_exposes_meeting_tools():
    tools = asyncio.run(mcp.list_tools())
    assert {t.name for t in tools} == {"search_meetings", "list_meetings", "get_action_items", "topic_timeline"}
    assert all(t.description for t in tools)
