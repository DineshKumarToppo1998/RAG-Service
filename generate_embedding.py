
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
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



generate_vector_store()