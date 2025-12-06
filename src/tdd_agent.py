import gradio as gr
import os
import subprocess
import tempfile
from typing import Dict, List, Literal, Annotated
from langchain_core.messages import AnyMessamge
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from rich.console import Console

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    groq_api_key="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",
    temperature=0,
)

console = Console()

class TDDState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    goal: str
    test_code: str = ""
    impl_code: str = ""
    combined_code: str = ""
    execution_result: str = ""
    error: str | None = None
    iteration: int = 0
    final_answer: str = ""


class TestSuiteGeneration(BaseModel):
    test_code: str = Field(description="Complete Python unittest code including imports and test class.")
    description: str = Field(description="Brief description of edge cases covered.")

class ImplementationGeneration(BaseModel):
    impl_code: str = Field(description="The functional Python code.")
    explanation: str

class FinalResponse(BaseModel):
    summary: str


def generate_tests(state: TDDState) -> Dict:
    """Step 1: Generate comprehensive unit tests."""
    print(f"--- Generating Tests for: {state.goal} ---")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior QA Engineer.
        Your task is to write a comprehensive Python `unittest` suite for the user's goal.

        Requirements:
        1. Import `unittest`.
        2. Create a test class inheriting from `unittest.TestCase`.
        3. Include standard cases.
        4. CRITICAL: Include EDGE CASES (boundary values, empty inputs, negative numbers, large numbers, type errors).
        5. DO NOT write the implementation function, just the tests. Assume the function is named appropriately (e.g., `solve` or based on context).
        6. Make sure the code is self-contained (imports included).
        7. Total test cases should atleast be 10
        """),
        ("human", "Goal: {goal}")
    ])

    chain = prompt | llm.with_structured_output(TestSuiteGeneration)
    result = chain.invoke({"goal": state.goal})

    return {"test_code": result.test_code}

def generate_implementation(state: TDDState) -> Dict:
    """Step 2 (or Retry): Generate code to pass the tests."""
    print(f"--- Generating Implementation (Iter {state.iteration}) ---")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Python Developer.
        Write the implementation code to pass the provided unit tests.

        Requirements:
        1. Write ONLY the function/class implementation.
        2. Do not include the tests (I already have them).
        3. Handle the edge cases defined in the tests.
        """),
        ("human", """Goal: {goal}

        Existing Tests:
        {test_code}

        Previous Execution Errors (if any):
        {error}
        """)
    ])

    chain = prompt | llm.with_structured_output(ImplementationGeneration)
    result = chain.invoke({
        "goal": state.goal,
        "test_code": state.test_code,
        "error": state.error or "None"
    })

    return {"impl_code": result.impl_code}

def execute_suite(state: TDDState) -> Dict:
    """Step 3: Combine and Execute."""
    print("--- Executing Test Suite ---")

    main_execution_block = """
if __name__ == '__main__':
    unittest.main()
"""

    full_code = f"{state.impl_code}\n\n# --- TESTS ---\n{state.test_code}\n{main_execution_block}"


    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(full_code)
        temp_file_name = tmp_file.name

    try:

        process = subprocess.run(
            ["python", temp_file_name],
            capture_output=True,
            text=True,
            timeout=10
        )

        output = process.stdout + "\n" + process.stderr
        return_code = process.returncode

    except subprocess.TimeoutExpired:
        output = "Execution Timed Out."
        return_code = 1
    except Exception as e:
        output = str(e)
        return_code = 1
    finally:
        os.remove(temp_file_name)

    return {
        "combined_code": full_code,
        "execution_result": output,
        "iteration": state.iteration + 1,
        # If return code is 0, tests passed.
        "error": output if return_code != 0 else None
    }

def decide_next_step(state: TDDState) -> Literal["generate_implementation", "synthesize_answer"]:
    """Step 4: Decide Loop."""


    if state.error is None:
        return "synthesize_answer"


    if state.iteration >= 3:
        return "synthesize_answer"

    return "generate_implementation"

def synthesize_answer(state: TDDState) -> Dict:
    """Step 5: Final Report."""
    status = "SUCCESS" if state.error is None else "FAILED/PARTIAL"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the coding task. State clearly if tests passed or failed."),
        ("human", """Goal: {goal}
        Status: {status}
        Test Output: {output}
        """)
    ])

    chain = prompt | llm.with_structured_output(FinalResponse)
    result = chain.invoke({
        "goal": state.goal,
        "status": status,
        "output": state.execution_result
    })

    return {"final_answer": result.summary}

builder = StateGraph(TDDState)

builder.add_node("generate_tests", generate_tests)
builder.add_node("generate_implementation", generate_implementation)
builder.add_node("execute_suite", execute_suite)
builder.add_node("synthesize_answer", synthesize_answer)

# Flow
builder.add_edge(START, "generate_tests")
builder.add_edge("generate_tests", "generate_implementation")
builder.add_edge("generate_implementation", "execute_suite")

builder.add_conditional_edges(
    "execute_suite",
    decide_next_step,
    {
        "generate_implementation": "generate_implementation",
        "synthesize_answer": "synthesize_answer"
    }
)
builder.add_edge("synthesize_answer", END)

tdd_graph = builder.compile()



def run_tdd_agent(goal: str):
    """Wrapper to run graph and format output for UI"""
    initial_state = TDDState(goal=goal)
    final_state = tdd_graph.invoke(initial_state)

    test_display = f"### Generated Test Suite\n```python\n{final_state['test_code']}\n```"


    impl_display = f"###  Implementation\n```python\n{final_state['impl_code']}\n```"


    logs = final_state['execution_result']

    if final_state['error'] is None:
        status_msg = " ALL TESTS PASSED"
    else:
        status_msg = " TESTS FAILED (Max retries reached)"

    final_report = f"# {status_msg}\n\n**Agent Summary:** {final_state['final_answer']}"

    return test_display, impl_display, logs, final_report

with gr.Blocks(title="TDD Code Agent") as demo:
    gr.Markdown("# A.C.E - Agentic Coding Engine ")
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

            out_logs = gr.Textbox(label="Terminal Output", lines=10, interactive=False)
         with gr.Column():
            gr.Markdown("### Final Status")
            out_final = gr.Markdown()

    run_btn.click(
        fn=run_tdd_agent,
        inputs=[goal_input],
        outputs=[out_tests, out_impl, out_logs, out_final]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)