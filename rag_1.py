import os
from pathlib import Path
from langchain import hub
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

def process_rag_system(input_message, session_id):

    ######################################################################################################################################
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "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"

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token= "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",
    )

    ######################################################################################################################################
    # Load, chunk and index the contents of the blog.
    loader = PyMuPDFLoader("main_doc.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    try:
        if splits:
            vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
        else:
            print("No valid splits found in the document.")
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return "Error processing the document."

    ######################################################################################################################################

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()


    ### Contextualize question ###
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
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ######################################################################################################################################
    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ######################################################################################################################################
    ### Statefully manage chat history ###
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    ######################################################################################################################################
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    response = conversational_rag_chain.invoke(
        {"input": input_message},
        config={
            "configurable": {"session_id": session_id}
        }
    )

    return response["answer"]

