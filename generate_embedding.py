import os, shutil
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



Settings.embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-2d-large-v1",device="cuda")
def generate_vector_store():
    docs = SimpleDirectoryReader(input_dir="./docs").load_data()
    # splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model)
    index = VectorStoreIndex.from_documents(documents=docs, transformations=[splitter],show_progress=True)
    index.storage_context.persist(persist_dir="./embeddings")
    print("Embeddings created for the documents Provided !!")
    clear_dir()

    

    
def clear_dir(directory='./docs'):
    for root, directories, files in os.walk(directory):
        for file in files:
            os.unlink(os.path.join(root, file))  # Delete file
        for directory in directories:
            shutil.rmtree(os.path.join(root, directory))
    print("Documents Deleted!!")



def update_vector_store(new_docs_dir="./docs", persist_dir="./embeddings"):
    
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context=storage_context)
    new_docs = SimpleDirectoryReader(input_dir=new_docs_dir).load_data()

    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model
    )
    for doc in new_docs:
        index.insert(doc, transformations=[splitter])

    index.storage_context.persist(persist_dir=persist_dir)
    clear_dir()
    

