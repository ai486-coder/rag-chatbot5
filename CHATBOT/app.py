import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import torch

# Flask App
app = Flask(__name__)
vectorstore = None
conversation_chain = None
chat_history = []

# Extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

# Create a FAISS vectorstore from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Initialize conversation chain
def get_conversation_chain(vectorstore):
    model_name = "microsoft/DialoGPT-medium"  # Free conversational model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        device=0 if torch.cuda.is_available() else -1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    # Custom prompt to avoid unwanted phrases
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Context:
{context}

Question: {question}
Answer:"""
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    return conversation_chain

# Home page: upload PDFs
@app.route('/')
def index():
    return render_template('index.html')

# Process PDFs and initialize chain
@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return redirect('/chat')

# Chat page
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global conversation_chain, chat_history

    if conversation_chain is None:
        return redirect('/')

    if request.method == 'POST':
        user_question = request.form['user_question']
        response = conversation_chain({'question': user_question})
        chat_history = [(msg.content, msg.type) for msg in response['chat_history']]

    return render_template('chat.html', chat_history=chat_history)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)