import os 
import streamlit as st 
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG Q&A Chatbot with PDF Uploads Including Chat History")
st.write("Upload PDF Files and Chat with their Content!!")

groq_api_key = st.text_input("Enter your Groq API Key: ", type="password")

if groq_api_key:
    llm=ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF File", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []

        for uploaded_file in uploaded_files:
            temp_pdf = f"./PDF Files/temp.pdf"

            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            pdf_loader = PyPDFLoader(temp_pdf)
            docs = pdf_loader.load()
            documents.extend(docs)

        text_spliiter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_spliiter.split_documents(documents=documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "Formulate a standalone question which can be understood "
            "without the chat history. Don't answer the question, "
            "Just reformulate it if needed and otherwise return it as it is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

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
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
                
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question: ")

        if user_input:
            session_history = get_session_history(session_id=session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            
            st.write(st.session_state.store)
            st.write("Assistant: ", response["answer"])
            st.write("Chat History: ", session_history.messages)

else:
    st.warning("Please Enter the Groq API Key!!")




