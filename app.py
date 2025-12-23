from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ---------------- APP INIT ----------------
app = Flask(__name__)

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- EMBEDDINGS ----------------
embeddings = download_hugging_face_embeddings()

# ---------------- PINECONE ----------------
index_name = "medibot-index"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ---------------- LLM ----------------
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# ---------------- RAG PIPELINE ----------------
rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]
    response = rag_chain.invoke(user_msg)
    return response


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
