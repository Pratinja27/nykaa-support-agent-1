from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.guardrails import apply_input_guards, groundedness_ok
from agent.memory import append_turn, get_session
from agent.schemas import validate_response
from agent.tools import check_order_status
from config import ACTIVE_COLLECTION, SIMILARITY_THRESHOLD
from mock_llm import classify_intent, extract_record_id, generate_from_context


class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    masked_query: str
    intent: str
    injection_blocked: bool
    record_id: Optional[str]
    context: str
    sources: List[str]
    order_info: Optional[Dict[str, Any]]
    answer: str
    grounded: bool
    response: Dict[str, Any]
    executed_nodes: List[str]


def _touch(state, name):
    done = list(state.get("executed_nodes") or [])
    done.append(name)
    return done


def guard_node(state: AgentState) -> AgentState:
    raw = state["query"]
    guarded = apply_input_guards(raw)
    return {
        "masked_query": guarded["masked_query"],
        "injection_blocked": guarded["injection"]["blocked"],
        "executed_nodes": _touch(state, "guard"),
    }


def classify_node(state: AgentState) -> AgentState:
    sess = get_session(state.get("session_id") or "default")
    last_id = sess.get("last_record_id")
    if state.get("injection_blocked"):
        return {
            "intent": "blocked",
            "record_id": None,
            "executed_nodes": _touch(state, "classify"),
        }
    intent = classify_intent(state["masked_query"], last_record_id=last_id)
    rid = extract_record_id(state["masked_query"], last_record_id=last_id)
    return {
        "intent": intent,
        "record_id": rid,
        "executed_nodes": _touch(state, "classify"),
    }


def route_intent(state: AgentState) -> str:
    if state.get("injection_blocked") or state.get("intent") == "blocked":
        return "format"
    if state.get("intent") == "order_status":
        return "order"
    return "rag"


def rag_node(state: AgentState) -> AgentState:
    from rag.retrieval import context_from_hits, retrieve

    strategy = "sentence" if ACTIVE_COLLECTION.endswith("sentence") else "fixed"
    hits = retrieve(state["masked_query"], strategy=strategy)
    top1 = hits[0]["similarity"] if hits else 0.0
    below = top1 < SIMILARITY_THRESHOLD
    context, sources, _kept = context_from_hits(hits)
    answer = generate_from_context(state["masked_query"], context, below_threshold=below)
    ok = groundedness_ok(answer, context)
    if not ok and "i don't know" not in answer.lower():
        answer = "I don't know. Retrieved context does not support this answer."
        ok = True
    return {
        "context": context,
        "sources": sources,
        "answer": answer,
        "grounded": ok,
        "order_info": None,
        "executed_nodes": _touch(state, "rag"),
    }


def order_node(state: AgentState) -> AgentState:
    info = check_order_status(state.get("record_id") or "")
    if not info.get("found"):
        answer = "I don't know. No order was found for that record id."
    else:
        extra = ""
        if info["escalate"]:
            extra = " Escalation score %.2f is at or above the threshold, so this should move to L2." % info["escalation_score"]
        else:
            extra = " Escalation score %.2f is below the threshold." % info["escalation_score"]
        answer = (
            "Order %s is %s. Order value is INR %s. Delayed shipment is %s. Days since created: %s.%s"
            % (
                info["record_id"],
                info["status"],
                info["order_value_inr"],
                info["delayed_shipment"],
                info["days_since_created"],
                extra,
            )
        )
    return {
        "order_info": info,
        "context": "",
        "sources": [],
        "answer": answer,
        "grounded": True,
        "executed_nodes": _touch(state, "order"),
    }


def format_node(state: AgentState) -> AgentState:
    if state.get("injection_blocked"):
        payload = {
            "answer": "Request blocked by the prompt-injection guardrail.",
            "intent": "blocked",
            "grounded": True,
            "sources": [],
            "record_id": None,
            "escalation_score": None,
            "escalate": None,
        }
    else:
        info = state.get("order_info") or {}
        payload = {
            "answer": state.get("answer") or "",
            "intent": state.get("intent") or "policy",
            "grounded": bool(state.get("grounded")),
            "sources": list(state.get("sources") or []),
            "record_id": info.get("record_id") if info else state.get("record_id"),
            "escalation_score": info.get("escalation_score") if info else None,
            "escalate": info.get("escalate") if info else None,
        }
    validate_response(payload)
    sid = state.get("session_id") or "default"
    append_turn(sid, "user", state.get("masked_query") or state.get("query") or "")
    rid = payload.get("record_id")
    append_turn(sid, "assistant", payload["answer"], record_id=rid)
    return {
        "response": payload,
        "executed_nodes": _touch(state, "format"),
    }


def build_graph(checkpointer=None, interrupt_after=None):
    builder = StateGraph(AgentState)
    builder.add_node("guard", guard_node)
    builder.add_node("classify", classify_node)
    builder.add_node("rag", rag_node)
    builder.add_node("order", order_node)
    builder.add_node("format", format_node)
    builder.add_edge(START, "guard")
    builder.add_edge("guard", "classify")
    builder.add_conditional_edges(
        "classify",
        route_intent,
        {"rag": "rag", "order": "order", "format": "format"},
    )
    builder.add_edge("rag", "format")
    builder.add_edge("order", "format")
    builder.add_edge("format", END)
    kwargs = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if interrupt_after:
        kwargs["interrupt_after"] = interrupt_after
    return builder.compile(**kwargs)


_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def run_agent(query, session_id="default"):
    app = get_app()
    state = app.invoke({"query": query, "session_id": session_id, "executed_nodes": []})
    return state["response"]
