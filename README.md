# A.C.E — Agentic Coding Engine

A fully open, quantized, multi-agent coding system built on StarCoder2 and LangGraph.

**A.C.E** (Agentic Coding Engine) is a modular, fully open-source agentic coding framework designed to explore whether resource-efficient, 4-bit quantized models can perform multi-step code reasoning, test generation, debugging, retrieval-augmented coding, and iterative repair.

The system combines LangGraph, StarCoder2, QLoRA fine-tuning, Python sandboxing, and an interactive Gradio UI to build a practical and reproducible agentic coding pipeline.

## Key Features

**Four agentic paradigms**
- REPL agent ([repl_agent.py](./src/repl_agent.py)): iterative generation + execution corrections
- TDD Tool ([tdd_agent.py](./src/tdd_agent.py)): test-first generation, then implementation
- Self-Correcting Tool ([self_correct_agent.py](./src/self_correct_agent.py)): repairs both tests and implementation
- RAG-powered Tool ([rag_agent.py](./src/rag_agent.py)): retrieves internal documentation using FAISS

**Model suite:**
- StarCoder2-3B (4-bit)
- StarCoder2-7B (4-bit)
- StarCoder2-7B-FT (QLoRA fine-tuned)
- LLaMA-4 Maverick-17B via Groq API

**Additional capabilities:**
- Quantized 4-bit execution for single-GPU compatibility
- Deterministic Python sandbox with isolated subprocess execution
- Gradio UI for interactive, transparent agent workflows
- HumanEval + MBPP evaluation pipeline
- Ablation studies across agentic components and model scales

## System Overview

A.C.E integrates LLM reasoning + execution feedback + retrieval + fine-tuning through an explicit LangGraph state machine.

### Architecture Components

**Model Layer:** StarCoder2 models (3B/7B) + finetuned 7B + Maverick-17B

**Agent Layer:** REPL, TDD, Self-Correcting, and RAG agents

**Execution Layer:**
- Deterministic Python sandbox
- Subprocess isolation
- AST validation and static safety checks

**Retrieval Layer:** Internal FAISS index of synthetic documentation

**UI Layer:** Gradio app for interactive logs, test visualization, and debugging

The system logs every decision, test, exception, and state transition using LangSmith for full reproducibility.

## 🔧 Installation

```bash
git clone https://https://github.com/Swag369/A.C.E.git
cd A.C.E
pip install -r requirements.txt
```

### Requirements (Core)

- Python 3.10+
- PyTorch + bitsandbytes
- transformers, accelerate, peft
- langchain, langgraph, langsmith
- gradio
- faiss-cpu
- groq (for Maverick-17B)

## Running the Agents

### 1. Launch the Gradio Interface

```bash
python starcoder2-7b_UI_logging.py
```

This opens an interactive dashboard where you can:
- Enter a coding task
- View generated tests, implementations, failures
- Inspect logs + state transitions
- Run the full A.C.E workflow end-to-end

### 2. Run the REPL Agent

```bash
python REPL_agent.py
```

Or use the notebook: `REPL_agent.ipynb`

### 3. Run QLoRA Fine-Tuning

All training code is provided in: `finetune_starcoder_7b.ipynb`

The fine-tuned adapter is automatically merged and evaluated.

## Evaluation

A.C.E is evaluated on:

**HumanEval**
- 164 algorithmic Python tasks
- Standard Pass@1 evaluation
- Deterministic execution + hidden tests

**Google MBPP (sanitized)**
- 974 beginner/intermediate problems
- Broader coverage of everyday Python tasks

| Model | HumanEval | MBPP |
|-------|-----------|------|
| StarCoder2-3B | 26% | 23% |
| StarCoder2-7B | 39% | 39% |
| StarCoder2-7B-FT | 47% | 48% |
| LLaMA-4-Maverick-17B | 69% | — |
| GPT-3.5 / GPT-4 | 77–84% | — |

Both benchmarks run through an identical CodeChain sandbox, ensuring consistent comparison.

## Ablation Studies

We evaluate the contribution of each agentic component:

- Static syntax + safety checks
- Execution feedback
- TDD
- Test repair
- RAG retrieval
- QLoRA fine-tuning

**Key findings:**
- TDD gives the largest non-finetuning improvement
- Execution-feedback fixes near-miss logic
- Test repair prevents dead-end failures
- RAG helps with API grounding
- Fine-tuning provides the single largest performance jump

A combined ablation table for 7B-FT and Maverick-17B is included in the paper.

## UI Features

The Gradio interface helps visualize:

- Generated tests
- Code drafts
- Debugging traces
- Loop iterations
- State transitions

## QLoRA Fine-Tuning Summary

Trained on GitHub-code-clean + MATH23K with the following configuration:

- LoRA rank 8–16
- Learning rate ≈ 2e-4
- Runs directly on the 4-bit quantized StarCoder2-7B
- Produces StarCoder2-7B-FT

Fine-tuning improves:

- Pass@1 performance
- Test quality
- Stability across agent loops
- Edge-case reasoning

## Future Extensions

- Multi-agent planner–coder–verifier workflows
- Integration of static analyzers, type checkers, symbolic execution
- Large-scale evaluation on SWE-bench, GitHub issues
- Semantic retrieval over real codebases
- Human-in-the-loop preference correction (dislike → iterative refinement)


## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Contact

For questions or feedback, please open an issue on GitHub or reach out to [hemanthn@umd.edu].
