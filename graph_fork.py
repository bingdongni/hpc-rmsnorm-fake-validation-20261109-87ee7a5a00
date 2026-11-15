import asyncio
from collections.abc import Sequence
from typing import Annotated

from typing_extensions import TypedDict
from langgraph.channels import DeltaChannel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

def concatenate(state: list, writes: Sequence[list]) -> list:
    return state + [item for write in writes for item in write]

class State(TypedDict):
    history: Annotated[list, DeltaChannel(concatenate)]

def compiled(saver):
    def approve(state):
        return {"history": [interrupt("choose a route")]}
    return (StateGraph(State).add_node("plan", lambda s: {"history": ["planned"]})
            .add_node("approve", approve).add_edge(START, "plan").add_edge("plan", "approve")
            .add_edge("approve", END).compile(checkpointer=saver))

source, fork = [{"configurable": {"thread_id": n}} for n in ("original", "alternative")]
with SqliteSaver.from_conn_string(":memory:") as saver:
    graph = compiled(saver)
    start = graph.invoke({"history": ["input"]}, source)
    assert start["__interrupt__"]
    saver.copy_thread("original", "alternative")
    assert graph.invoke(Command(resume="alternate"), fork)["history"] == ["input", "planned", "alternate"]
    assert graph.get_state(source).next == ("approve",)
    assert graph.invoke(Command(resume="original"), source)["history"] == ["input", "planned", "original"]
    assert graph.get_state(fork).values["history"][-1] == "alternate"

async def async_fork():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = compiled(saver)
        start = await graph.ainvoke({"history": ["input"]}, source)
        assert start["__interrupt__"]
        await asyncio.to_thread(saver.copy_thread, "original", "alternative")
        assert (await graph.ainvoke(Command(resume="alternate"), fork))["history"] == ["input", "planned", "alternate"]
        assert (await graph.aget_state(source)).next == ("approve",)
        assert (await graph.ainvoke(Command(resume="original"), source))["history"] == ["input", "planned", "original"]
        assert (await graph.aget_state(fork)).values["history"][-1] == "alternate"

asyncio.run(async_fork())
print("SQLite sync and async graphs both resumed two divergent interrupt paths with DeltaChannel history")
