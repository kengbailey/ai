import json
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm
import pandas as pd




# Prepare train/eval data 
TRAIN_FILES = ["./Scaling_Laws_for_Downstream_Task_Performance_of_Large_Language_Models.pdf"] # TODO: pass in from argument or config file 
VAL_FILES = ["./Unraveling_the_Mystery_of_Scaling_Laws.pdf"] # TODO: pass in from argument or config file

TRAIN_CORPUS_FPATH = "./train_corpus.json" # TODO: pass in from argument or config file
VAL_CORPUS_FPATH = "./val_corpus.json" # TODO: pass in from argument or config file

def load_corpus(files):
    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=True)
    print(f"Parsed {len(nodes)} nodes")
    return nodes

train_nodes = load_corpus(TRAIN_FILES)
val_nodes = load_corpus(VAL_FILES)


# Generate synthetic Q/A pairs 
quantization_conf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"
    return prompt

def huggingface_llm(model_name="HuggingFaceH4/zephyr-7b-beta",
                    tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
                    context_window=3900,
                    max_new_tokens=256,
                    quantization_config = quantization_conf
                   ):
    llm = HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        model_kwargs={"quantization_config": quantization_config},
        # tokenizer_kwargs={},
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        messages_to_prompt=messages_to_prompt,
        device_map="auto",
    )

    return llm

llm = huggingface_llm()

train_dataset = generate_qa_embedding_pairs(
    llm=llm, nodes=train_nodes, verbose=False
)
val_dataset = generate_qa_embedding_pairs(
    llm=llm, nodes=val_nodes, verbose=False
)

train_dataset.save_json("train_dataset.json") # TODO: pass in from argument or config file 
val_dataset.save_json("val_dataset.json") # TODO: pass in from argument or config file 


# Finetune embedding model 
finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en", # TODO: pass in model id from argument or config file 
    model_output_path="bge_ft_SL", # TODO: pass in output path from argument or config file
    val_dataset=val_dataset,
)

finetune_engine.finetune()

embed_model = finetune_engine.get_finetuned_model()

# Evaluate embedding model 
def evaluate(
    dataset,
    embed_model,
    top_k=5,
    verbose=False,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(
        nodes, embed_model=embed_model, show_progress=True
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    return eval_results

ada = OpenAIEmbedding(api_key='sk-') # TODO: get from os env
ada_val_results = evaluate(val_dataset, ada)
df_ada = pd.DataFrame(ada_val_results)
hit_rate_ada = df_ada["is_hit"].mean()
hit_rate_ada 

bge = "local:BAAI/bge-small-en" # TODO: get model name from argument 
bge_val_results = evaluate(train_dataset, bge)
df_bge = pd.DataFrame(bge_val_results)
hit_rate_bge = df_bge["is_hit"].mean()
hit_rate_bge

val_results_finetuned = evaluate(val_dataset, embed_model)
df_finetuned = pd.DataFrame(val_results_finetuned)
hit_rate_finetuned = df_finetuned["is_hit"].mean()
hit_rate_finetuned

print(f"Hit rate for Ada: {hit_rate_ada}")
print(f"Hit rate for BGE: {hit_rate_bge}")
print(f"Hit rate for Finetuned: {hit_rate_finetuned}")
