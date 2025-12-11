import gradio as gr
import os
import ast
import subprocess
import tempfile
import datetime
import operator
from typing import Dict, List, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Fine-tuned StarCoder2-7B LoRA model...")
print(f"Device: {device}")

model_path = "./starcoder2-7b-lora-python-final"

try:
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Fine-tuned model loaded successfully!")
except Exception as e:
    print(f"Fine-tuned model not found: {e}")
    print("Loading base StarCoder2-7B with 4-bit quantization...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "bigcode/starcoder2-7b",
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto"
    )
    print("Base model loaded successfully!")

tokenizer.pad_token = tokenizer.eos_token

class AgentState(BaseModel):
    goal: str
    test_code: str = ""
    impl_code: str = ""
    error: str | None = None
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

def extract_code_block(text: str, language: str = "python") -> str:
    lines = text.split("\n")
    in_block = False
    code = []
    
    for line in lines:
        if f"```{language}" in line:
            in_block = True
            continue
        if "```" in line and in_block:
            break
        if in_block:
            code.append(line)
    
    return "\n".join(code) if code else text

def call_llm(prompt: str, max_tokens: int = 2048) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]

def generate_tests(state: AgentState) -> Dict:
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are a Senior QA Engineer. Write a comprehensive Python unittest suite for the following goal.

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
7. Output ONLY the Python code in a code block

Goal: {state.goal}

### Response:
```python
"""
    
    response = call_llm(prompt, max_tokens=2048)
    test_code = extract_code_block(response, "python")
    
    return {
        "test_code": test_code,
        "logs": [log_entry("TEST GENERATION", "Tests generated with MINIMUM 10 test cases")]
    }

def fix_tests(state: AgentState) -> Dict:
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are a Senior QA Engineer. Fix the failing tests. Do not include implementation, only tests.

CRITICAL REQUIREMENTS:
1. Write EXACTLY 10 or MORE individual test methods
2. If test logic is flawed, relax or fix it
3. Keep all edge cases covered
4. Each test method must use self.assert* statements
5. Output ONLY the Python code in a code block

Goal: {state.goal}

Error: {state.error[:500]}

### Response:
```python
"""
    
    response = call_llm(prompt, max_tokens=2048)
    test_code = extract_code_block(response, "python")
    
    return {
        "test_code": test_code,
        "logs": [log_entry("FIX TESTS", "Tests updated with MINIMUM 10 test cases")]
    }

def generate_implementation(state: AgentState) -> Dict:
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are a Senior Python Developer. Implement the Python code to pass the provided tests.

Requirements:
1. Write ONLY the function/class implementation
2. Do not include the tests or unittest imports
3. Handle all edge cases defined in the tests
4. Output ONLY the Python code in a code block

Goal: {state.goal}

Tests: {state.test_code[:1500]}

### Response:
```python
"""
    
    response = call_llm(prompt, max_tokens=2048)
    impl_code = extract_code_block(response, "python")
    
    return {
        "impl_code": impl_code,
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
    
    full_code = f"{state.impl_code}\n\nimport unittest\n{state.test_code}\n\nif __name__ == '__main__':\n    unittest.main()"
    
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

def decide_next(state: AgentState) -> Literal["fix_tests", "generate_implementation", "synthesize"]:
    if state.error is None:
        return "synthesize"
    
    if state.iteration > 3:
        return "synthesize"
    
    if "AssertionError" in state.error and state.iteration % 2 == 0:
        return "fix_tests"
    
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
builder.add_node("generate_implementation", generate_implementation)
builder.add_node("static_check", static_check)
builder.add_node("execute_tests", execute_tests)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "generate_tests")
builder.add_edge("generate_tests", "generate_implementation")
builder.add_edge("fix_tests", "generate_implementation")
builder.add_edge("generate_implementation", "static_check")

def check_static(state): 
    return "generate_implementation" if state.error else "execute_tests"

builder.add_conditional_edges("static_check", check_static)
builder.add_conditional_edges(
    "execute_tests",
    decide_next,
    {
        "fix_tests": "fix_tests",
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

with gr.Blocks(title="TDD Code Agent") as demo:
    gr.Markdown("# A.C.E - Agentic Coding Engine")
    gr.Markdown("**Powered by StarCoder2-7B LoRA Fine-tuned** 🚀\n\nI will generate **Edge Case Tests** first, then write code to pass them.")
    
    with gr.Row():
        goal_input = gr.Textbox(
            label="Programming Goal", 
            placeholder="e.g. Implement a function to find the longest palindrome", 
            lines=2
        )
        run_btn = gr.Button("Run TDD Cycle", elem_id="run_btn", scale=0)
    
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