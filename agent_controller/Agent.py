from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub


llm = HuggingFaceHub(
    repo_id="bigcode/starcoder2-7b",
    model_kwargs={"temperature": 0.1, "max_length": 512}
)


prompt = PromptTemplate.from_template("Using the following context: {context}\n Help get the following code working: {code_snippet}")

final_prompt = prompt.format(context="This is the context information.", code_snippet="def example_function(): pass")
response = llm.invoke(final_prompt)

parser = StrOutputParser.from_llm(llm, expected_output=YourExpectedOutputType)


chain = prompt | llm | parser

response = chain.invoke({})

print(response)