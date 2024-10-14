import streamlit as st
from rag_logic import RAG_PDF



def load_sidebar():
    """
    Load the sidebar with the file uploader and the process button
    """
    with st.sidebar:
        pdfFile = st.file_uploader("Upload a PDF file", type=["pdf", "csv", "txt"])
        if st.button("Process the files"):
            if pdfFile.name.split(".")[-1] == "pdf":
                with st.spinner("Loading PDF file..."):
                    rag = RAG_PDF(pdfFile)
                    rag.get_pdf_text()
                    rag.get_text_chunk()
                    rag.get_text_embeddings()
                    rag.load_llm()
                    
                st.success("PDF file loaded successfully")
            elif pdfFile.name.split(".")[-1] == "csv":
                st.error("Please upload a PDF file")
            elif pdfFile.name.split(".")[-1] == "txt":
                st.error("Please upload a PDF file")

def main():
    """
    Main function of the app. Here we call the functions to load the sidebar and the chatbot
    We also initialize the chat history
    We also use the RAG_PDF class to load the pdf file and the LLM model

    """
    global rag_logic
    rag_logic = RAG_PDF()
    st.set_page_config(page_title="SampleSet Chatbot", page_icon=":books:")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title('SampleSet Chatbot')


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "context":
            with st.expander("Context"):
                st.markdown(message["content"])
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask you question related to the pdf?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            response = rag_logic.prompt_template_text_response(prompt)

        with st.expander("Context"):
            st.markdown(response["context"])

        st.session_state.messages.append({"role": "context", "content": response["context"]})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    load_sidebar()

if __name__ == "__main__":
    main()