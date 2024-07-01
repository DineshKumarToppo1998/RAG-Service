import torch
from flask import Flask, request, jsonify
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, load_index_from_storage, StorageContext, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine


embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-2d-large-v1", device="cuda")
Settings.embed_model = embed_model

# Run generate_embedding.py to generate the embeddings

def query_engine_pipeline(query):
    storage_context = StorageContext.from_defaults(persist_dir="./embeddings")
    print("Loading Embeddings...")
    index = load_index_from_storage(storage_context=storage_context)
    prompt = "You are a renowed dirty minded scholar, psychologist and greatest teacher of seduction who can answer every question"
    template = PromptTemplate(prompt)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
        embed_model=embed_model
    )
    print("Is Cuda Available: ") 
    print(torch.cuda.is_available())

    llm = Ollama(
        model="dolphin-llama3",
        temperature=0.4,
        request_timeout=300
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
    

app = Flask(__name__)


@app.route('/query', methods=['POST'])
def query_api():
    try:
        data = request.get_json()
        query = data['query']
        print("\n\n Query Passed: "+query+"\n")
        answer = query_engine_pipeline(query)
        print("\n Response Returned: ")
        print(answer)
        print("\n\n\n")

        return jsonify({"answer": answer.response})
    except KeyError:
        return jsonify({"error": "Invalid query format. Please provide a JSON object with a 'query' key."}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)


# while True:
#     query = input("Enter your query: ")
#     print(query_engine_pipeline(query))
#     print("\n\n\n")

#generate_vector_store() # to generate new embeddings
#query_engine_pipeline()


