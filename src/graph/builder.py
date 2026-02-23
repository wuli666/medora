from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    planner_node,
    reflector_node,
    summarize_node,
    supervisor_node,
    tooler_node,
)
from src.graph.state import MedAgentState


def build_graph():
    graph = StateGraph(MedAgentState)

    # --- nodes ---
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tooler", tooler_node)
    graph.add_node("planner", planner_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("summarize", summarize_node)

    # --- edges ---
    graph.add_edge(START, "supervisor")

    # Planner-centric loop: workers return to planner
    graph.add_edge("tooler", "planner")
    graph.add_edge("reflector", "planner")

    # Summarize terminates the graph
    graph.add_edge("summarize", END)

    # Supervisor uses Command for dynamic routing — no outgoing edges needed

    return graph.compile()


# Lazy singleton
_app = None


def get_graph_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
