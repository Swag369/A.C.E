

#Loading data for RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = TextLoader("agent_controller/Agent.py") # lets make this get python documentation actually
documents = loader.load()

#split chunks based on
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80) #split chunks on hierarchical whitespace 
chunks = text_splitter.split_documents(documents)



embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectordb = Chroma(
    collection_name="dev_docs",
    embedding_function=embedder,
    persist_directory="./chroma_store"
)
vectorstore = vectordb.from_documents(chunks, embedder)
vectordb.persist() #does this mean if I start my program twice, I repopulate my DB unnecessarily?


# retrieved_docs = vectorstore.similarity_search(query, k=3)