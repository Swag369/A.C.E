import os
import itertools
from typing import Annotated, Dict, List, Literal

from datasets import load_dataset


def modify_test(problem):
  def add_try_except(assertions, curr_assert):
    assertions.append(f"{lead_space}try:")
    assertions.append(f"{lead_space*2}cnt_total += 1")
    assertions.append("\n".join([f"{lead_space}{line}" for line in curr_assert]))
    assertions.append(f"{lead_space*2}cnt_correct += 1")
    assertions.append(f"{lead_space}except Exception as e:")
    assertions.append(f"{lead_space*2}pass")
    return []

  lines = problem['test'].split("\n")
  lines = [line for line in lines if len(line.strip()) > 0 and (not line.strip().startswith('#'))]
  new_lines = []
  keyword = "assert"

  for j, line in enumerate(lines):
    new_lines.append(line)
    if "def check(candidate):" in lines[j]:
      lead_space = ''.join(itertools.takewhile(str.isspace, lines[j+1]))
      lead_indent = len(lead_space)
      new_lines.append(f"{lead_space}cnt_correct, cnt_total = 0, 0")
      break

  assertions = []
  curr_assert = []
  for i in range(j+1, len(lines)):
    line = lines[i]
    lead_indent = 0

    if keyword in line:
      if len(curr_assert) > 0:
        curr_assert = add_try_except(assertions, curr_assert)

    curr_assert.append(line)

  if len(curr_assert) > 0:
    curr_assert = add_try_except(assertions, curr_assert)

  new_lines.extend(assertions)
  new_lines.append(f"{lead_space}return cnt_correct, cnt_total")

  new_lines.append(f"\ndef get_result():\n{lead_space}return check({problem['entry_point']})")
  problem['new_test'] = "\n".join(new_lines)

def write_code_to_file(problem, result: dict):
    task_id = problem['task_id']

    # Create directory if it doesn't exist
    directory_path = os.path.dirname(task_id)
    os.makedirs(directory_path, exist_ok=True)
    file_path = task_id + '.py'

    modify_test(problem)

    test_cases = problem['new_test']
    code_to_write = [result["code_to_execute"], test_cases]

    with open(file_path, "w") as f:
        f.write('\n'.join(code_to_write))

    print(f"Code successfully written to {file_path}")

def prep_prompt(code: str, prompt: str, example: str, indent=2):
  res = ["\n\n"]
  lines = code.strip().split("\n")
  indent_space = ' '*indent
  example = example.replace("assert", "").strip()
  for line in lines:
    if line.startswith("def"):
      res.append(line)
      res.append(f'{indent_space}"""')
      res.append(f'{indent_space}{prompt}')
      res.append(f'{indent_space}>>> {example}')
      res.append(f'{indent_space}"""')
      return '\n'.join(res)
  return f"{prompt}\n{code}"

def prep_test(test_list: list, indent=2):
  res = ["\n\ndef check(candidate):"]
  indent_space = ' '*indent
  for test in test_list:
    if "check(" in test: continue # prevent recursive call (e.g ID=56)
    res.append(f'{indent_space}{test}')
  return '\n'.join(res)

def prep_entry_point(code: str):
  lines = code.strip().split("\n")
  for line in lines:
    if line.startswith("def"):
      res = line.replace("def", "").strip()
      res = res[:res.index("(")]
      return res
  return ""

def reformat_mbpp_to_humaneval(row, name="mbpp"):
  row['task_id'] =  f"{name}/{row['task_id']}"
  row['prompt'] = prep_prompt(code=row['code'], prompt=row['prompt'], example=row['test_list'][0])
  row['test'] = prep_test(test_list=row['test_list'])
  row['entry_point'] = prep_entry_point(code=row['code'])
  row['canonical_solution'] = row['code']
  del row['code'], row['test_list']
  return row

def view_dataset(dataset, idx=0):
  for k in dataset[idx].keys():
    print(k)
    print(dataset[idx][k])
    print('-'*50)

def run_inference(dataset):
  test_len = len(dataset)
  for i in range(test_len):
    problem = dataset[i]
    try:
      result = graph.invoke(REPLState(goal=problem["prompt"]))
      write_code_to_file(problem, result)
    except Exception as e:
      print(f"Error in {i}-th problem --> skip: {e}")
    # if i==5: break


if __name__ == "__main__":
    # Load the entire dataset
    humaneval_dataset = load_dataset("openai/openai_humaneval")
    humaneval_dataset_test = humaneval_dataset["test"]

    # mbpp_dataset = load_dataset("google-research-datasets/mbpp")
    # mbpp_dataset_test = mbpp_dataset["test"]

    mbpp_dataset_sanitized = load_dataset("google-research-datasets/mbpp", "sanitized")
    mbpp_dataset_sanitized_test = mbpp_dataset_sanitized["test"]

    mbpp_dataset_sanitized_test_clean = mbpp_dataset_sanitized_test.map(reformat_mbpp_to_humaneval, fn_kwargs={"name": "mbpp"})

    len(humaneval_dataset_test), len(mbpp_dataset_sanitized_test_clean)

    view_dataset(mbpp_dataset_sanitized_test_clean, idx=0)
    view_dataset(humaneval_dataset_test, idx=123)

    run_inference(dataset=humaneval_dataset_test)
    run_inference(dataset=mbpp_dataset_sanitized_test_clean)
