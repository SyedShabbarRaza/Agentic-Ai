# import os
# import fitz
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# load_dotenv()
# from sentence_transformers import SentenceTransformer
# from sentence_transformers import util

# llm=ChatGroq(
#     model="llama-3.3-70b-versatile",
#     api_key=os.getenv("Groq_Api_Key")
# )

# model = SentenceTransformer("all-MiniLM-L6-v2")

# doc = fitz.open("shabbar_raza_resume.pdf")

# text = ""

# for page in doc:
#     text += page.get_text()

# chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
# # chunk.strip() extra spaces tabs wgera ko remove kry ga
# # text.split() jaha bhi double new line hoga wahan se split kr dy ga yaani paragraphs ko split kr dy ga

# print(f"Total chunks: {len(chunks)}")
# data_embeddings = model.encode(chunks)

# def retrieve(question: str):

#     question_embedding = model.encode(question)

#     scores = util.cos_sim(question_embedding, data_embeddings)
#     k = min(3, len(chunks))
     
#     if k == 0:
#          return "No context available."
         
#     top_indices = scores[0].topk(k=k).indices.tolist()

#     context = "\n\n".join(chunks[index] for index in top_indices)

#     return context
    
# question=input("Question: ")
# context=retrieve(question)


# prompt = f"""
# You are a helpful assistant.

# Answer the question using the context but in your own wordings.

# Context:
# {context}

# Question:
# {question}

# If the answer is not present in the context, say:
# "I don't have enough information."
# """

# answer=llm.invoke(prompt)


# print(answer)





# Production style RAG

# 1.) Changing the text.split() method because it can't handle 
# badly written pdfs like no proper spacings or too long paragraphs.

# 2.) Using FAISS instead of cosine similarity because it is more efficient for large datasets.



# import os
# import fitz
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# load_dotenv()
# from sentence_transformers import SentenceTransformer,util
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEmbeddings

# def load_pdf(file_name):
#     text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=50,
#     chunk_overlap=10
#     )
#     doc = fitz.open(f"{file_name}.pdf")

#     text = ""
#     for page in doc:
#        text += page.get_text()

#     chunks = text_splitter.split_text(text)
#     print(f"Total chunks: {len(chunks)}")
#     documents = [Document(page_content=chunk) for chunk in chunks]
#     return documents

# llm=ChatGroq(
#     model="llama-3.3-70b-versatile",
#     api_key=os.getenv("Groq_Api_Key")
# )

# embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )



# # vector_store = FAISS.from_documents(
# #     load_pdf("shabbar_raza_resume"),
# #     embeddings
# # )

# vector_store = FAISS.load_local(
#     "resume_index",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# # Save to local disk for future use
# vector_store.save_local("resume_index")

# def retrieve(question):
#     docs = vector_store.similarity_search(question, k=3)
#     context = "\n\n".join(doc.page_content for doc in docs)

#     return context
    
# question=input("Question: ")
# context=retrieve(question)


# prompt = f"""
# You are a helpful assistant.

# Answer the question using the context but in your own wordings.

# Context:
# {context}

# Question:
# {question}

# If the answer is not present in the context, say:
# "I don't have enough information."
# """

# answer=llm.invoke(prompt)


# print(answer)





import os
import fitz
from dotenv import load_dotenv
load_dotenv()

# ---------------- LANGCHAIN ----------------
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- LANGGRAPH ----------------
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
# -------------------------------------------


# ---------------- PDF LOADING ----------------
def load_pdf(file_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10
    )

    doc = fitz.open(f"{file_name}.pdf")

    text = ""
    for page in doc:
        text += page.get_text()

    chunks = text_splitter.split_text(text)

    documents = [Document(page_content=chunk) for chunk in chunks]

    return documents


# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("Groq_Api_Key")
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Create once
# vector_store = FAISS.from_documents(
#     load_pdf("shabbar_raza_resume"),
#     embeddings
# )
# vector_store.save_local("resume_index")

# Load every time
vector_store = FAISS.load_local(
    "resume_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# ---------------- RETRIEVER ----------------
def retrieve(question):
    docs = vector_store.similarity_search(question, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    return context


# ---------------- LANGGRAPH STATE ----------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------- NODE ----------------
def chatbot(state: State):

    # Latest user message
    question = state["messages"][-1].content

    context = retrieve(question)

    prompt = f"""
You are a helpful assistant.

Conversation History:
{state["messages"]}

Context:
{context}

Question:
{question}

Answer using BOTH the conversation history and the retrieved context.

If the answer is not present, say:
"I don't have enough information."
"""

    answer = llm.invoke(prompt)

    return {
        "messages": [answer]
    }


# ---------------- GRAPH ----------------
builder = StateGraph(State)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(
    checkpointer=MemorySaver()
)


# ---------------- CHAT LOOP ----------------
config = {
    "configurable": {
        "thread_id": "resume_chat"
    }
}

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    response = graph.invoke(
        {
            "messages": [
                ("user", question)
            ]
        },
        config=config
    )

    print("\nAI:", response["messages"][-1].content)

