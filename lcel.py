from langchain_openai import AzureChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import AzureAISearchRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


import os

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
    service_name=endpoint,api_key=api_key, content_key="content", top_k=1, index_name=index_name
)

def create_history_aware(retriever):
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

def create_question_answer_chain():

        system_prompt = (
            """
            You are an onboarding assistant. Here's the context from company documents: {context}
            """
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        return create_stuff_documents_chain(llm, qa_prompt)

def create_retrieval():
    return create_retrieval_chain(create_history_aware(retriever), create_question_answer_chain())

def create_conversational_rag_chain(session_id):
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]
    return RunnableWithMessageHistory(
        create_retrieval(),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

response = create_conversational_rag_chain('testMedium').invoke(
       {"input": "who is the company"},
        config={"configurable": {"session_id": 'testMecdium'}}
)

print(response["answer"])