from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, load_index_from_storage, StorageContext, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine


embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-2d-large-v1", device="cuda")
Settings.embed_model = embed_model
def generate_vector_store():
    docs = SimpleDirectoryReader(input_dir="./docs").load_data()
    splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    index = VectorStoreIndex.from_documents(documents=docs, transformations=[splitter],show_progress=True)
    index.storage_context.persist(persist_dir="./embeddings")


def query_engine_pipeline(query):
    storage_context = StorageContext.from_defaults(persist_dir="./embeddings")
    index = load_index_from_storage(storage_context=storage_context)
    prompt = "You are a renowed scholar. Please read the provided books and answer questions on it."
    template = PromptTemplate(prompt)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        embed_model=embed_model
    )
    llm = Ollama(
        model="dolphin-llama3",
        temperature=0.5,
        request_timeout=120.0
    )
    #prompt_2 = template.format(context_str = retriever)
    respSynth = get_response_synthesizer(
        llm=llm,
        response_mode="tree_summarize",
        text_qa_template=template,
        use_async=True,
        verbose=True
    )
    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=respSynth
    )
    answer = engine.query(query)
    return answer
    
while True:
    query = input("Enter your query: ")
    print(query_engine_pipeline(query))

#generate_vector_store() # to generate new embeddings
#query_engine_pipeline()