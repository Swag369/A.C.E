from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: List[BaseMessage]
    plan: Optional[List[str]]
    retrieved_docs: Optional[List[str]]
    code_snippet: Optional[str]
    error_trace: Optional[str]


def planner_node(state: AgentState) -> AgentState:
    # read state["messages"], call a model, produce state["plan"]
    return new_state

def coder_node(state: AgentState) -> AgentState:
    # read state["messages"], call a model, produce state["plan"]
    return new_state

def debugger_node(state: AgentState) -> AgentState:
    # read state["messages"], call a model, produce state["plan"]
    return new_state

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("debugger", debugger_node)

workflow.set_entry_point("planner")

#how to add different possible edges
#does it auto route on those edges?
#what about optional tool calls?

workflow.add_edge("planner", "coder")
workflow.add_edge("planner", "debugger")

workflow.add_edge("coder", "planner")
workflow.add_edge("coder", "debugger")

workflow.add_edge("debugger", "planner")
workflow.add_edge("debugger", END)

graph = workflow.compile()