import gradio as gr
import os
import subprocess
import tempfile
from typing import Dict, List, Literal, Annotated
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from rich.console import Console

# --- 1. SETUP & MODELS ---

# Initialize LLM
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    groq_api_key="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",
    temperature=0,
)

console = Console()

# Define State
class REPLState(BaseModel):
    action: Literal["continue", "complete"] = Field(default="continue")
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    code_history: List[str] = Field(default_factory=list)
    code_to_execute: str = Field(default="")
    last_execution_result: str | None = None
    error: str | None = None
    goal: str | None = None
    final_answer: str | None = None

# Define Structured Outputs
class REPLDecision(BaseModel):
    action: Literal["continue", "complete"]
    rationale: str

class CodeGeneration(BaseModel):
    code: str
    explanation: str

class FinalSynthesis(BaseModel):
    answer: str
    explanation: str

# --- 2. DEFINE NODES ---

def decide_action(state: REPLState) -> Dict:
    """Decide whether to continue execution or complete."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Python coding assistant. Choose 'complete' if the goal is achieved and output is correct. Choose 'continue' if more steps are needed."),
        ("human", "Goal: {goal}\n\nLast result: {last_result}\nHistory len: {history_len}\n\nShould we continue or complete?")
    ])

    chain = prompt | llm.with_structured_output(REPLDecision)
    result = chain.invoke({
        "goal": state.goal,
        "last_result": state.last_execution_result or "No execution yet",
        "history_len": len(state.code_history)
    })
    return {"action": result.action}

def generate_code(state: REPLState) -> Dict:
    """Generate Python code."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate complete, self-contained Python code to achieve the goal. Include print statements to show results."),
        ("human", "Goal: {goal}\n\nPrevious result: {last_result}")
    ])

    chain = prompt | llm.with_structured_output(CodeGeneration)
    result = chain.invoke({
        "goal": state.goal,
        "last_result": state.last_execution_result or "No previous execution",
    })
    return {"code_to_execute": result.code}

def execute_code(state: REPLState) -> Dict:
    """Execute code using subprocess."""
    code = state.code_to_execute

    # Write to temp file and execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(code)
        temp_file_name = tmp_file.name

    try:
        process_result = subprocess.run(
            ["python", temp_file_name],
            capture_output=True,
            text=True,
        )
        # Capture stdout and stderr
        output = process_result.stdout
        if process_result.stderr:
            output += f"\n[Stderr]: {process_result.stderr}"

        success = True
    except Exception as e:
        output = str(e)
        success = False
    finally:
        os.remove(temp_file_name)

    return {
        "last_execution_result": output,
        "code_history": state.code_history + [code],
        "error": None if success else output,
    }

def synthesize_answer(state: REPLState) -> Dict:
    """Final summary."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Create a clear answer based on the execution results."),
        ("human", "Goal: {goal}\n\nFinal Result: {final_result}")
    ])

    chain = prompt | llm.with_structured_output(FinalSynthesis)
    result = chain.invoke({
        "goal": state.goal,
        "final_result": state.last_execution_result,
    })
    return {"final_answer": result.answer}

# --- 3. BUILD GRAPH ---

builder = StateGraph(REPLState)
builder.add_node("decide_action", decide_action)
builder.add_node("generate_code", generate_code)
builder.add_node("execute_code", execute_code)
builder.add_node("synthesize_answer", synthesize_answer)

builder.add_edge(START, "decide_action")
builder.add_conditional_edges(
    "decide_action",
    lambda state: state.action,
    {"continue": "generate_code", "complete": "synthesize_answer"},
)
builder.add_edge("generate_code", "execute_code")
builder.add_edge("execute_code", "decide_action")
builder.add_edge("synthesize_answer", END)

graph = builder.compile()

# --- 4. GRADIO UI (Fixed) ---

def run_agent_ui(goal: str):
    """Handler for the UI"""
    initial_state = REPLState(goal=goal)
    final_state = graph.invoke(initial_state)

    # Format History
    history_log = ""
    for i, code in enumerate(final_state["code_history"]):
        history_log += f"--- STEP {i+1} ---\n"
        history_log += f"[CODE]:\n{code}\n"
        history_log += "\n"

    history_log += f"--- FINAL EXECUTION OUTPUT ---\n{final_state['last_execution_result']}"

    return final_state['final_answer'], history_log

# Build UI without the theme argument to avoid version conflicts
with gr.Blocks(title="Python REPL Agent") as demo:
    gr.Markdown("## 🐍 Python Code Agent")
    gr.Markdown("Enter a goal. The agent will write code, execute it, and give you the answer.")

    with gr.Row():
        with gr.Column():
            goal_input = gr.Textbox(label="Goal", placeholder="Calculate the square root of 12345", lines=2)
            run_btn = gr.Button("Run Agent", variant="primary")
        with gr.Column():
            result_output = gr.Textbox(label="Final Answer", lines=4)

    logs_output = gr.Code(label="Execution History (Code & Output)", language="python")

    run_btn.click(fn=run_agent_ui, inputs=goal_input, outputs=[result_output, logs_output])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)