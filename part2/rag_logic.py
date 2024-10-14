from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
import faiss
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

class RAG_PDF:
    def __init__(self, pdf_file = ""):
        self.store = {}
        self.pdf_file = pdf_file
        self.text = ""
        self.user_prompt = ""
        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use ten sentences maximum and keep the "
            "answer detailed."
            "\n\n"
            "{context}"
            )
        self.embeddings = GPT4AllEmbeddings()
        
        self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    ("human", "{input}"),
                ]
            )

    def get_pdf_text(self):
        if not self.pdf_file:
            raise ValueError("Please provide a pdf file")
        else:
            pdf_reader = PdfReader(self.pdf_file)
            
            for page in pdf_reader.pages:
                self.text += page.extract_text()
        # return self.text
    
    def get_text_chunk(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splits = text_splitter.split_text(self.text)

    def get_text_embeddings(self):
        # self.embeddings = GPT4AllEmbeddings()
        index = faiss.IndexFlatL2(len(self.embeddings.embed_query("Hello World")))

        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.vector_store.add_texts(texts=self.splits)
        self.vector_store.save_local("vector_store")
        self.retriever = self.vector_store.as_retriever()


    def load_llm(self):
        self.llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="AIzaSyCnsg5BXgC-NgBPw2mWoKAG8-GBJF4MyX8")

    def context_aware_retrieval(self):
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )


        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use ten sentences maximum and keep the "
            "answer detailed."
            "\n\n"
            "{context}"
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

    def get_session_history(self, session_id: str = "") -> BaseChatMessageHistory:
        if session_id not in self.store.keys():
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def chat_history(self):
        
        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def prompt_template_text_response(self, user_prompt):
        if not os.path.exists("vector_store") or not os.path.exists("vector_store/index.faiss") or not os.path.exists("vector_store/index.pkl"):
            self.get_pdf_text()
            self.get_text_chunk()
            self.get_text_embeddings()
            self.load_llm()
        else:
            self.vector_store = FAISS.load_local("vector_store", self.embeddings, allow_dangerous_deserialization=True)
            self.retriever = self.vector_store.as_retriever()
            self.load_llm()
            self.context_aware_retrieval()
            self.chat_history()

        self.user_prompt = user_prompt
        # question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        # rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        # results = rag_chain.invoke({"input": self.user_prompt})
        results = self.conversational_rag_chain.invoke(
            {"input": self.user_prompt},
            config={
                "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )
        return results
    

def main():
    rag_pdf = RAG_PDF(pdf_file="Airport_Rules_Regs_7_27_22.pdf")
    # rag_pdf.get_pdf_text()
    # rag_pdf.get_text_chunk()
    # rag_pdf.get_text_embeddings()
    # prompt="What is the use of baggage carts?", 
    # rag_pdf.load_llm()
    response = rag_pdf.prompt_template_text_response(user_prompt="What is the use of baggage carts?")
    print(response)

if __name__ == "__main__":
    main()