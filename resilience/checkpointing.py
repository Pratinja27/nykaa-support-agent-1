import os
import sqlite3
import time

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from typing import List, TypedDict

from config import CHECKPOINT_PATH, TRANSCRIPT_DIR

EXEC_LOG = os.path.join(TRANSCRIPT_DIR, "checkpoint_exec.log")


class CkptState(TypedDict, total=False):
    query: str
    seen: List[str]
    answer: str


def _mark(name):
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    with open(EXEC_LOG, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (name, time.time()))


def intake(state: CkptState) -> CkptState:
    _mark("intake")
    return {"seen": list(state.get("seen") or []) + ["intake"]}


def route(state: CkptState) -> CkptState:
    _mark("route")
    return {"seen": list(state.get("seen") or []) + ["route"]}


def lookup(state: CkptState) -> CkptState:
    _mark("lookup")
    return {"seen": list(state.get("seen") or []) + ["lookup"]}


def reply(state: CkptState) -> CkptState:
    _mark("reply")
    return {
        "seen": list(state.get("seen") or []) + ["reply"],
        "answer": "done for %s" % state.get("query"),
    }


def build_ckpt_graph(checkpointer):
    g = StateGraph(CkptState)
    g.add_node("intake", intake)
    g.add_node("route", route)
    g.add_node("lookup", lookup)
    g.add_node("reply", reply)
    g.add_edge(START, "intake")
    g.add_edge("intake", "route")
    g.add_edge("route", "lookup")
    g.add_edge("lookup", "reply")
    g.add_edge("reply", END)
    return g.compile(checkpointer=checkpointer, interrupt_after=["route"])


def read_exec_log():
    if not os.path.exists(EXEC_LOG):
        return []
    with open(EXEC_LOG, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def run_checkpoint_demo(thread_id="nykaa-ckpt-1"):
    if os.path.exists(EXEC_LOG):
        os.remove(EXEC_LOG)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    app = build_ckpt_graph(saver)
    config = {"configurable": {"thread_id": thread_id}}

    first = app.invoke({"query": "status of NYK-1001", "seen": []}, config)
    after_interrupt = read_exec_log()
    snap = app.get_state(config)

    second = app.invoke(None, config)
    after_resume = read_exec_log()

    names_first = [ln.split()[0] for ln in after_interrupt]
    names_all = [ln.split()[0] for ln in after_resume]
    rerun = names_all.count("intake") > 1 or names_all.count("route") > 1

    return {
        "thread_id": thread_id,
        "checkpoint_db": CHECKPOINT_PATH,
        "after_interrupt_nodes": names_first,
        "after_resume_nodes": names_all,
        "interrupt_state_seen": first.get("seen"),
        "final_seen": second.get("seen"),
        "final_answer": second.get("answer"),
        "next_after_interrupt": list(snap.next) if snap else [],
        "completed_nodes_reexecuted": rerun,
        "exec_log": after_resume,
    }
