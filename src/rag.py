import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

def create_rag_chain(markdown_path):
    # Load the markdown file
    loader = UnstructuredMarkdownLoader(markdown_path)
    document = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a Chroma vector store from the document splits
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Define the prompt template
    prompt_template = """
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    I will tip you $1000 if the user finds the answer helpful.
    <context>
    {context}
    </context>

    Question: {input}
    """

    # Create the retriever from the vector store
    retriever = vectorstore.as_retriever()

    # Initialize the language model
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.0,
        # additional params
    )

    # Create the prompt from the template
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Function to format the documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


