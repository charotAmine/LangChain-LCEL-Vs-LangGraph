from langgraph.graph import StateGraph, END

from typing import Dict, TypedDict
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict
from IPython.display import Image, display

# Azure Cognitive Search details
endpoint = ""  # Replace with your Azure search endpoint
api_key = ""    # Replace with your Azure search API key
index_name = ""  # Define your index name
store = {}

os.environ["OPENAI_API_VERSION"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""


# Initialize LLM (Azure OpenAI)
llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment="gpt-4o",
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["OPENAI_API_VERSION"],
)

retriever = AzureAISearchRetriever(
    service_name=endpoint,api_key=api_key, content_key="content", top_k=5, index_name=index_name
)

def retrieve(state):
    print("RETRIEVE NODE")
    state_dict = state["keys"]
    question = state_dict["question"]
    local = state_dict["local"]
    documents = retriever.invoke(question)
    return {"keys": {"documents": documents, "local": local, 
            "question": question}}

def format_docs(state):
    print("FORMAT DOCUMENT NODE")

    state_dict = state["keys"]
    question = state_dict["question"]
    local = state_dict["local"]
    documents = state_dict["documents"]

    document = "\n\n".join(doc.page_content for doc in documents)
    return {"keys": {"formatted_documents": document, "local": local, 
            "question": question}}

def generate(state):
    print("generate:", state)
    state_dict = state["keys"]
    question = state_dict["question"]
    formatted_docs = state_dict["formatted_documents"]
    result = chain_with_prompt.invoke({"question": question, "context": formatted_docs})
    return {"keys": {"formatted_documents": formatted_docs, "response": result, 
            "question": question}}

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain_with_prompt = prompt | llm | StrOutputParser()

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


# Build LangGraph workflow
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("format_document", format_docs)  # format documents
workflow.add_node("generate", generate)  # generatae

workflow.add_edge("retrieve", "format_document")
workflow.add_edge("format_document", "generate")
workflow.add_edge("generate", END)

workflow.set_entry_point("retrieve")

app = workflow.compile()

try:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
except:
    pass

# Employee asks a question
employee_query = "What is the company’s policy on remote work?"

result = app.invoke({"keys": { "local": "", 
            "question": employee_query}})

print("Response : ",result['keys']['response'])
