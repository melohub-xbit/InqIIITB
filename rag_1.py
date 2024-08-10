<<<<<<< HEAD
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

import logging

logging.basicConfig(filename='rag_system.log', level=logging.INFO)
print("Starting RAG system...")
print("This is outside the call function")

def process_rag_system(input_message, session_id):
    print("Starting RAG system...")

    ########################################################################################################
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "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"

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token= "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",
    )

    print("Set up the llm")
    ########################################################################################################
    # Load, chunk and index the contents of the blog.
    loader = PyMuPDFLoader("main_doc.pdf")
    docs = loader.load()

    print("now splitting text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("storing to chroma")
    try:
        print("In the try statement")
        if splits:
            print("In the if statement")
            vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
            print("Vectorstore created successfully")
        else:
            print("No valid splits found in the document.")
    except Exception as e:
        print("In the except block")
        print(f"Error creating vectorstore: {e}")
        return "Error processing the document."

    ########################################################################################################
    print("docs loaded")

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()

    print("retriever is set")

    print("RAG system initialized.")    ### Contextualize question ###
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
    print("retriever and prompts set")


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
    ########################################################################################################
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    print("conv rag chain has been set")

    print(f"Processing input: {input_message} for session: {session_id}")

    # print("Welcome to InqIIITB! What do you want to know about?")

    # user_prompt = "START"

    # while (user_prompt != "END" and user_prompt != "BYE"):
    #     user_ques = input("Your question here: ")
    #     response = conversational_rag_chain.invoke(
    #         {"input": user_ques},
    #         config={
    #             "configurable": {"session_id": "abc123"}
    #         },  # constructs a key "abc123" in `store`.
    #     )
    #     print("Chat History:", get_session_history("abc123").messages)
    #     print("Response:", response["answer"])
    #     # print(response["answer"])
    #     user_prompt = input("Enter END or BYE to end the conversation, or any thing else to continue: ")
    #     if (user_prompt == "END" or user_prompt == "BYE"):
    #         print("Thank you! Hope I answered your queries. Bye!")
    print("invoking the chain")
    response = conversational_rag_chain.invoke(
        {"input": input_message},
        config={
            "configurable": {"session_id": session_id}
        }
    )
    print(f"Generated Response:")
    print(response["answer"])
    print("RAG system initialized.")

    return response["answer"]
    # return f"Processed: {input_message}"

=======
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
from langchain_core.messages import AIMessage, HumanMessage

########################################################################################################

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token= "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",
)


########################################################################################################
# Load, chunk and index the contents of the blog.
loader = PyMuPDFLoader(
    "main_doc.pdf"
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

if splits:
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
else:
    print("No valid splits found in the document.")
########################################################################################################

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

########################################################################################################


######################################################################################################################################
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# # 2. Incorporate the retriever into a question-answering chain.
# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise. And if the user says something where they "
#     "mean to end the conversation, conclude the conversation then politely."
#     "\n\n"
#     "{context}"
# )

# contextualize_q_system_prompt = (
#     "Given a chat history and the latest user question "
#     "which might reference context in the chat history, "
#     "formulate a standalone question which can be understood "
#     "without the chat history. Do NOT answer the question, "
#     "just reformulate it if needed and otherwise return it as is."
# )

# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )


# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

######################################################################################################################################


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
########################################################################################################
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# for chunk in rag_chain.stream({"input":"Tell me about TEDxIIITBangalore?"}):
#     print(chunk, end="\n", flush=True)
# response = rag_chain.invoke({"input": "Tell me about LeanIN club"})

# response = conversational_rag_chain.invoke(
#     {"input": "What is SquareOne?"},
#     config={
#         "configurable": {"session_id": "abc123"}
#     },  # constructs a key "abc123" in `store`.
# )
# print("###############################################################################################")
# print(response["answer"])
# print("###############################################################################################")
# r2 = conversational_rag_chain.invoke(
#     {"input": "Who are part of it's team?"},
#     config={"configurable": {"session_id": "abc123"}},
# )["answer"]
# print("###############################################################################################")
# print(r2)
# print("###############################################################################################")

print("Welcome to InqIIITB! What do you want to know about?")

user_prompt = "START"
while (user_prompt != "END" and user_prompt != "BYE"):
    user_ques = input("Your question here: ")
    response = conversational_rag_chain.invoke(
        {"input": user_ques},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )
    print("Chat History:", get_session_history("abc123").messages)
    print("Response:", response["answer"])
    # print(response["answer"])
    user_prompt = input("Enter END or BYE to end the conversation, or any thing else to continue: ")
    if (user_prompt == "END" or user_prompt == "BYE"):
        print("Thank you! Hope I answered your queries. Bye!")
>>>>>>> f716a4b37cc7ee1598b2aaf37726d6dfdb93cf8a
