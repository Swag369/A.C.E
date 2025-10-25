import os, re, json, sys, tempfile, uuid, shutil, subprocess
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage



MAX_INTERACTIONS_BEFORE_FINISHING_TASK = 20


class Workspace:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else Path(tempfile.gettempdir()) / f"agent_ws_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
    def path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError("Path escapes workspace")
        return p
    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)

WS = Workspace()
MAX_FILE_BYTES = 200_000
ALLOWED_EXTS = {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml"}
PY_TIMEOUT = 12


#file manipulation tools

def _read(p: Path) -> str:
    b = p.read_bytes()
    if len(b) > MAX_FILE_BYTES: raise ValueError("File too large")
    return b.decode("utf-8", errors="replace")

def _write(p: Path, s: str):
    b = s.encode("utf-8")
    if len(b) > MAX_FILE_BYTES: raise ValueError("Too large")
    if p.suffix and p.suffix not in ALLOWED_EXTS: raise ValueError(f"Ext {p.suffix} not allowed")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)

def _run(cmd: List[str], cwd: Path, timeout_s: int = PY_TIMEOUT) -> str:
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        return "[timeout] exceeded"
    return f"[exit={proc.returncode}]\nSTDOUT:\n{out}\nSTDERR:\n{err}"



#file manipulation tool wrappers
def write_file(path: str, content: str) -> str:
    _write(WS.path(path), content)
    return f"wrote {path} ({len(content)} bytes)"

def read_file(path: str) -> str:
    p = WS.path(path)
    return _read(p) if p.exists() else "not found"

def list_dir(path: str = ".") -> str:
    p = WS.path(path)
    if not p.exists(): return "dir not found"
    if not p.is_dir(): return "not a dir"
    items = [{"name": c.name, "type": "dir" if c.is_dir() else "file", "bytes": c.stat().st_size} for c in sorted(p.iterdir())]
    return json.dumps(items, indent=2)

def run_python(code: Optional[str] = None, file: Optional[str] = None, args: Optional[List[str]] = None) -> str:
    args = args or []
    if code:
        _write(WS.path("main.py"), code)
        target = WS.path("main.py")
    elif file:
        target = WS.path(file)
        if not target.exists(): return "file not found"
    else:
        return "need `code` or `file`"
    return _run([sys.executable, str(target), *args], cwd=WS.root)

def run_tests(path: str = ".") -> str:
    target = WS.path(path)
    if not target.exists(): return "path not found"
    return _run([sys.executable, "-m", "pytest", str(target), "-q", "--maxfail=1"], cwd=WS.root, timeout_s=60)


# ---------- ReAct prompt ----------
SYSTEM = """You are a careful coding agent.
You have tools: write_file, read_file, list_dir, run_python, run_tests.
Rules:
- Think in small steps. Inspect workspace often with list_dir.
- Prefer tests and running code frequently.
- NEVER access paths outside the workspace.
- When you want to use a tool, reply ONLY in this exact schema:

Action: <one of write_file|read_file|list_dir|run_python|run_tests>
Action Input: <a single JSON object with the tool arguments>

When you are done, reply with:
Final Answer: <your final result/summary>
"""

#parser for LLM output
ACTION_RE = re.compile(r"Action:\s*(?P<tool>\w+)\s*[\r\n]+Action Input:\s*(?P<input>\{.*\})", re.DOTALL)

def step(model, messages):
    resp = model.invoke(messages).content
    m = ACTION_RE.search(resp)
    if not m:
        # Look for Final Answer
        if "Final Answer:" in resp:
            return resp, True
        # Ask model to format correctly
        messages.append(HumanMessage(content="Format error. Use the exact Action/Action Input or Final Answer schema."))
        return resp, False
    tool, raw = m.group("tool"), m.group("input")
    try:
        payload = json.loads(raw)
    except Exception as e:
        messages.append(HumanMessage(content=f"Your JSON was invalid: {e}. Please reformat."))
        return resp, False

    # Dispatch
    if tool == "write_file": out = write_file(**payload)
    elif tool == "read_file": out = read_file(**payload)
    elif tool == "list_dir": out = list_dir(**payload)
    elif tool == "run_python": out = run_python(**payload)
    elif tool == "run_tests": out = run_tests(**payload)
    else: out = f"unknown tool {tool}"

    # Feed result back
    tool_report = f"Tool Result:\n{out}\n\n(continue; choose another Action or produce Final Answer)"
    messages.append(HumanMessage(content=tool_report))
    return resp, False



if __name__ == "__main__":
    # Use quantized StarCoder2 via Ollama
    llm = ChatOllama(model="starcoder2:3b-q3_K_M", temperature=0.1)  # try 3b for speed

    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            "Task: Create mathops.py with fib(n) and tests/test_mathops.py with pytest tests; "
            "iterate until tests pass; finally show the directory tree."
        )),
        HumanMessage(content=f"(Workspace root is {WS.root}) Start by listing the directory.")
    ]

    done = False
    for _ in range(MAX_INTERACTIONS_BEFORE_FINISHING_TASK):  # safety cap
        _, done = step(llm, messages)
        if done: break

    print("\n--- Conversation ---")
    for m in messages: print(type(m).__name__, ">>", m.content)
    print(f"\nWorkspace: {WS.root}")