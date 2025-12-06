import gradio as gr
import os
import ast
import subprocess
import tempfile
import datetimem
from typing import Dict, List, Literal, Annotated
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    groq_api_key="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",
    temperature=0,
)

search_tool = DuckDuckGoSearchRun()

class AgentState(BaseModel):
    goal: str
    test_code: str = ""
    impl_code: str = ""
    error: str | None = None
    search_results: str = ""
    iteration: int = 0
    final_answer: str = ""
    logs: Annotated[List[str], operator.add] = Field(default_factory=list)

class TestSuite(BaseModel):
    test_code: str = Field(description="Complete Python unittest code.")

class Implementation(BaseModel):
    impl_code: str = Field(description="The functional Python code.")


def log_entry(step_name: str, content: str) -> str:
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    return f"\n{'='*20}\n[{timestamp}] STEP: {step_name}\n{'='*20}\n{content}\n"



def generate_tests(state: AgentState) -> Dict:
    """Step 1: Generate Tests."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a comprehensive Python `unittest` suite for the user's goal. Include Edge Cases. Minimum of 10 testcases"),
        ("human", "Goal: {goal}")
    ])
    res = (prompt | llm.with_structured_output(TestSuite)).invoke({"goal": state.goal})
    return {
        "test_code": res.test_code,
        "logs": [log_entry("TEST GENERATION", res.test_code[:300] + "...")]
    }

def fix_tests(state: AgentState) -> Dict:

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a QA Engineer. The tests you wrote are failing the implementation. "
                   "If the test logic is flawed (e.g., too strict on non-deterministic outputs), relax or fix the test case. minimum of 10 testcases"),
        ("human", """Goal: {goal}
        Current Tests: {test_code}
        Implementation: {impl_code}
        Error: {error}

        Rewrite the tests to be correct and robust.""")
    ])
    res = (prompt | llm.with_structured_output(TestSuite)).invoke({
        "goal": state.goal,
        "test_code": state.test_code,
        "impl_code": state.impl_code,
        "error": state.error
    })
    return {
        "test_code": res.test_code,
        "logs": [log_entry("FIX TESTS", "Tests updated based on execution error.")]
    }

def retrieval_node(state: AgentState) -> Dict:

    query = f"python {state.goal} library documentation"
    if state.error and ("ImportError" in state.error or "ModuleNotFoundError" in state.error):
        try:
            results = search_tool.invoke(query)
            return {
                "search_results": results,
                "logs": [log_entry("RETRIEVAL", f"Found: {results[:200]}...")]
            }
        except:
            pass
    return {"logs": [log_entry("RETRIEVAL", "Skipped")]}

def generate_implementation(state: AgentState) -> Dict:

    context = f"Docs: {state.search_results}\nError: {state.error}" if state.error else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Implement the Python code to pass the tests."),
        ("human", "Goal: {goal}\nTests: {test_code}\nContext: {context}")
    ])
    res = (prompt | llm.with_structured_output(Implementation)).invoke({
        "goal": state.goal, "test_code": state.test_code, "context": context
    })
    return {
        "impl_code": res.impl_code,
        "logs": [log_entry("CODE GENERATION", res.impl_code[:300] + "...")]
    }

def static_check(state: AgentState) -> Dict:

    try:
        ast.parse(state.impl_code)
        return {"error": None, "logs": [log_entry("STATIC CHECK", "Passed")]}
    except SyntaxError as e:
        return {"error": str(e), "logs": [log_entry("STATIC CHECK", f"Failed: {e}")]}

def execute_tests(state: AgentState) -> Dict:

    if state.error: return {"logs": [log_entry("EXECUTION", "Skipped due to Static Error")]}

    full_code = f"{state.impl_code}\n\n{state.test_code}\n\nif __name__ == '__main__':\n    unittest.main()"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(full_code)
        fname = tmp.name

    try:
        proc = subprocess.run(["python", fname], capture_output=True, text=True, timeout=10)
        output = proc.stdout + "\n" + proc.stderr
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        output = "Timeout"
        return_code = 1
    finally:
        os.remove(fname)

    return {
        "error": output if return_code != 0 else None,
        "iteration": state.iteration + 1,
        "logs": [log_entry("EXECUTION", f"Result Code: {return_code}\nOutput: {output[:500]}")]
    }

def decide_next(state: AgentState) -> Literal["fix_tests", "retrieval", "generate_implementation", "synthesize"]:

    if state.error is None:
        return "synthesize"

    if state.iteration > 3:
        return "synthesize"

    if "AssertionError" in state.error and state.iteration % 2 == 0:
         return "fix_tests"

    # 2. If Import Error, search
    if "ImportError" in state.error:
        return "retrieval"

    # 3. Default: Fix Code
    return "generate_implementation"

def synthesize(state: AgentState) -> Dict:
    status = "Pass" if state.error is None else "Fail"
    return {
        "final_answer": f"Status: {status}",
        "logs": [log_entry("FINISHED", f"Final Status: {status}")]
    }

# --- GRAPH ---
builder = StateGraph(AgentState)
builder.add_node("generate_tests", generate_tests)
builder.add_node("fix_tests", fix_tests) # New Node
builder.add_node("retrieval", retrieval_node)
builder.add_node("generate_implementation", generate_implementation)
builder.add_node("static_check", static_check)
builder.add_node("execute_tests", execute_tests)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "generate_tests")
builder.add_edge("generate_tests", "generate_implementation")
builder.add_edge("fix_tests", "generate_implementation")
builder.add_edge("retrieval", "generate_implementation")
builder.add_edge("generate_implementation", "static_check")

def check_static(state): return "generate_implementation" if state.error else "execute_tests"
builder.add_conditional_edges("static_check", check_static)

builder.add_conditional_edges(
    "execute_tests",
    decide_next,
    {
        "fix_tests": "fix_tests",
        "retrieval": "retrieval",
        "generate_implementation": "generate_implementation",
        "synthesize": "synthesize"
    }
)
builder.add_edge("synthesize", END)
graph = builder.compile()

# --- UI ---
def run_agent(goal):
    res = graph.invoke(AgentState(goal=goal))
    full_log = "".join(res["logs"])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(full_log)
        path = f.name
    return res['test_code'], res['impl_code'], full_log, path, res['final_answer']

with gr.Blocks(title="Self-Correcting Agent") as demo:
    gr.Markdown("# A.C.E - Agentic Coding Engine")
    goal = gr.Textbox(label="Goal")
    btn = gr.Button("Run")
    with gr.Row():
        tc = gr.Code(label="Tests")
        ic = gr.Code(label="Implementation")
    logs = gr.Textbox(label="Logs", lines=10)
    dl = gr.File(label="Download Logs")
    stat = gr.Markdown()
    btn.click(run_agent, inputs=goal, outputs=[tc, ic, logs, dl, stat])


if __name__ == "__main__":
    demo.launch(share=True, debug=True)
