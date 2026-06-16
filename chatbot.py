from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

print("Loading PDF...")
pdf_reader = PdfReader("sample.pdf")
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

print("Processing...")
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = splitter.split_text(text)

print("Creating search index...")
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(chunks, embeddings)

print("Setting up chatbot...")
llm = ChatOpenAI(model="gpt-3.5-turbo")
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

print("\n✓ Ready! Ask questions about your PDF.\n")

while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break
    try:
        docs = retriever.invoke(question)
        context = docs[0].page_content[:800]
        response = llm.invoke(f"Based on this:\n{context}\n\nAnswer: {question}")
        print(f"\nAnswer: {response.content}\n")
    except Exception as e:
        print(f"Error: {str(e)[:100]}\n")