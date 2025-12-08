import gradio as gr
import os
import ast
import subprocess
import tempfile
import datetime
import operator
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
    execution_output: str = ""
    logs: Annotated[List[str], operator.add] = Field(default_factory=list)

class TestSuite(BaseModel):
    test_code: str = Field(description="Complete Python unittest code with minimum 10 test cases.")

class Implementation(BaseModel):
    impl_code: str = Field(description="The functional Python code.")

def log_entry(step_name: str, content: str) -> str:
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] {step_name}\n{content}\n"

def generate_tests(state: AgentState) -> Dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior QA Engineer. Write a comprehensive Python `unittest` suite for the user's goal.

CRITICAL REQUIREMENTS:
1. Import unittest at the top
2. Create a test class inheriting from unittest.TestCase
3. Write EXACTLY 10 or MORE individual test methods (def test_*):
   - Standard cases
   - Edge cases (empty, single element, boundary values)
   - Negative numbers, zeros, large values
   - Duplicates
   - Type variations if applicable
4. Each test method should use self.assert* statements
5. DO NOT include the implementation, only tests
6. Make code self-contained

Count your test methods before submitting. MINIMUM 10 test methods."""),
        ("human", "Goal: {goal}")
    ])
    res = (prompt | llm.with_structured_output(TestSuite)).invoke({"goal": state.goal})
    return {
        "test_code": res.test_code,
        "logs": [log_entry("TEST GENERATION", "Tests generated with MINIMUM 10 test cases")]
    }

def fix_tests(state: AgentState) -> Dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior QA Engineer. The tests you wrote are failing the implementation. Fix the tests.

CRITICAL REQUIREMENTS:
1. Write EXACTLY 10 or MORE individual test methods
2. If test logic is flawed (too strict on non-deterministic outputs), relax or fix it
3. Keep all edge cases covered
4. Each test method must use self.assert* statements
5. Count your test methods: MINIMUM 10 required."""),
        ("human", "Goal: {goal}\nCurrent Tests: {test_code}\nImplementation: {impl_code}\nError: {error}\n\nRewrite to be correct and robust with MINIMUM 10 test methods.")
    ])
    res = (prompt | llm.with_structured_output(TestSuite)).invoke({
        "goal": state.goal,
        "test_code": state.test_code,
        "impl_code": state.impl_code,
        "error": state.error
    })
    return {
        "test_code": res.test_code,
        "logs": [log_entry("FIX TESTS", "Tests updated with MINIMUM 10 test cases")]
    }

def retrieval_node(state: AgentState) -> Dict:
    query = f"python {state.goal} library documentation"
    if state.error and ("ImportError" in state.error or "ModuleNotFoundError" in state.error):
        try:
            results = search_tool.invoke(query)
            return {
                "search_results": results,
                "logs": [log_entry("RETRIEVAL", "Documentation retrieved")]
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
        "goal": state.goal,
        "test_code": state.test_code,
        "context": context
    })
    return {
        "impl_code": res.impl_code,
        "logs": [log_entry("CODE GENERATION", "Implementation generated")]
    }

def static_check(state: AgentState) -> Dict:
    try:
        ast.parse(state.impl_code)
        return {"error": None, "logs": [log_entry("STATIC CHECK", "Syntax valid")]}
    except SyntaxError as e:
        return {"error": str(e), "logs": [log_entry("STATIC CHECK", f"Syntax Error: {e}")]}

def execute_tests(state: AgentState) -> Dict:
    if state.error:
        return {"logs": [log_entry("EXECUTION", "Skipped due to static error")]}

    full_code = f"{state.impl_code}\n\n{state.test_code}\n\nif __name__ == '__main__':\n    unittest.main()"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(full_code)
        fname = tmp.name
    try:
        proc = subprocess.run(["python", fname], capture_output=True, text=True, timeout=10)
        output = proc.stdout + "\n" + proc.stderr
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        output = "TIMEOUT: Execution took too long"
        return_code = 1
    finally:
        os.remove(fname)

    exec_log = log_entry("EXECUTION OUTPUT", output)

    return {
        "execution_output": output,
        "error": output if return_code != 0 else None,
        "iteration": state.iteration + 1,
        "logs": [exec_log]
    }

def decide_next(state: AgentState) -> Literal["fix_tests", "retrieval", "generate_implementation", "synthesize"]:
    if state.error is None:
        return "synthesize"

    if state.iteration > 3:
        return "synthesize"

    if "AssertionError" in state.error and state.iteration % 2 == 0:
        return "fix_tests"

    if "ImportError" in state.error or "ModuleNotFoundError" in state.error:
        return "retrieval"

    return "generate_implementation"

def synthesize(state: AgentState) -> Dict:
    status = "PASSED ✓" if state.error is None else "FAILED (Max retries)"
    summary = f"Status: {status}\nIterations: {state.iteration}"
    return {
        "final_answer": summary,
        "logs": [log_entry("FINISHED", summary)]
    }

builder = StateGraph(AgentState)
builder.add_node("generate_tests", generate_tests)
builder.add_node("fix_tests", fix_tests)
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

def check_static(state):
    return "generate_implementation" if state.error else "execute_tests"

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

log_file_path = None

def run_tdd_agent(goal: str):
    global log_file_path

    state = AgentState(goal=goal)
    final_state = graph.invoke(state)

    test_display = f"### Generated Test Suite\n```python\n{final_state['test_code']}\n```"
    impl_display = f"### Implementation\n```python\n{final_state['impl_code']}\n```"

    full_logs = "\n".join(final_state['logs'])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(full_logs)
        log_file_path = f.name

    status_icon = "✓ PASSED" if final_state['error'] is None else "✗ FAILED"
    final_report = f"# {status_icon}\n\nIterations: {final_state['iteration']}"

    return test_display, impl_display, full_logs, final_report, log_file_path

with gr.Blocks(title="ACE FULL") as demo:
    gr.Markdown("# A.C.E - Agentic Coding Engine")
    gr.Markdown("I will generate **Edge Case Tests** first, then write code to pass them.")

    with gr.Row():
        goal_input = gr.Textbox(label="Programming Goal", placeholder="e.g. Implement a function to find the longest palindrome", lines=2)
        run_btn = gr.Button("Run TDD Cycle", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Tests")
            out_tests = gr.Markdown()
        with gr.Column(scale=1):
            gr.Markdown("### Step 2: Solution")
            out_impl = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Execution Output (unittest)")
            out_logs = gr.Textbox(label="Terminal Output", lines=12, interactive=False)
        with gr.Column():
            gr.Markdown("### Final Status")
            out_final = gr.Markdown()

    with gr.Row():
        dl_logs = gr.File(label="Download Logs")

    run_btn.click(
        fn=run_tdd_agent,
        inputs=[goal_input],
        outputs=[out_tests, out_impl, out_logs, out_final, dl_logs]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)