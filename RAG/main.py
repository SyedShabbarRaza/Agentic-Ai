
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("Groq_Api_Key")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("knowledge.txt", "r") as file:
    knowledge = file.read()

chunks = knowledge.split("\n\n")

def retrieve(question:str):

    question_embedding = model.encode(question)
    
    for chunk in chunks:
        chunk_embedding = model.encode(chunk)
        score = util.cos_sim(question_embedding, chunk_embedding)
        print(f"Score for chunk: {score}")
        if score > 0.07:  # Adjust threshold as needed
            return chunk
    return "I don't know"
    
question=input("Question: ")
context=retrieve(question)


prompt = f"""
You are a helpful assistant.

Answer the question using the context but in your own wordings.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't have enough information."
"""

answer=llm.invoke(prompt)


print(answer)