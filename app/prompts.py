from app.models import User
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from dotenv import load_dotenv

# Load environment variables (specifically for the Gemini API key)
load_dotenv()
api_key = os.getenv('GEMINI_API')

# Initialize the large language model for conversational AI
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# Load PDF document
def uploadAndSplitPdfFile(f):
    file = "app\data\Evolution_of_AI.pdf"
    loader = PyPDFLoader(file)
    return loader

# Create the chunks
def createChunks(loader, size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    pages = loader.load_and_split(text_splitter)
    return pages

# Generate Embeddings
def generateEmbeddings(model):
    embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
    return embeddings

# Generate context from user data
def generate_context(user: User):
    context = f"""
    User Profile:
    ID: {user.id}
    Username: {user.username}
    Email: {user.email}
    Age: {user.age}
    Fitness Level: {user.level.value}
    """
    return context

# Define a prompt template to guide AI responses with a specific structure
def promptTemplate(template_text, pages):
    prompt = PromptTemplate.from_template(template_text)
    
    # Create a document chain that combines retrieved chunks for more contextual responses
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # Create the final retrieval chain that integrates retrieval and response generation
    vectordb = Chroma.from_documents(pages, generateEmbeddings("models/embedding-001")) 
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

# Write Questions (Prompts) and get response
def getPromptsAndReturnResponse(prompt, context, retrieval_chain):
    response = retrieval_chain.invoke({"input": prompt, "context": context})
    return response["answer"]
