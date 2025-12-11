from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


#setup LLM

llm = ChatOllama(model="starcoder2:7b", temperature=0.1)
resp = llm.invoke([
  SystemMessage(content="You reply with exactly 'pong'."),
  HumanMessage(content="ping")
])
print(resp.content)  # -> pong


#basic prompt
messages = [
  SystemMessage(content="You speak the Action/Final schema only."),
  HumanMessage(content="Start by listing the directory.")
]
resp = llm.invoke(messages)
messages.append(resp)


SYSTEM_PROMPT = """
                    You have tools: write_file, read_file, list_dir, run_python, run_tests.
                    Use ONLY these schemas:

                    Action: <tool>
                    Action Input: <JSON>

                    or

                    Final Answer: <text>

                """

#once LLM output parsed, execute tool
if tool == "write_file": out = write_file(**payload)
elif tool == "read_file": out = read_file(**payload)
...
else: out = f"unknown tool {tool}"

messages.append(HumanMessage(
  content=f"Tool Result:\n{out}\n\n(continue; choose another Action or produce Final Answer)"
))