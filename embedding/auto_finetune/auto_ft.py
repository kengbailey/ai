from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def load_corpus(files):
    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=True)
    print(f"Parsed {len(nodes)} nodes")
    return nodes

def prepare_documents(TRAIN_FILES, EVAL_FILES):
    train_nodes = load_corpus(TRAIN_FILES)
    eval_nodes = load_corpus(EVAL_FILES)
    return train_nodes, eval_nodes

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
                    max_new_tokens=256
                   ):
    quantization_conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    llm = HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        model_kwargs={"quantization_config": quantization_conf},
        # tokenizer_kwargs={},
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        messages_to_prompt=messages_to_prompt,
        device_map="auto",
    )

    return llm

def generate_qa_pairs(train_nodes, eval_nodes):
    llm = huggingface_llm()

    train_dataset = generate_qa_embedding_pairs(
        llm=llm, nodes=train_nodes, verbose=False
    )
    eval_dataset = generate_qa_embedding_pairs(
        llm=llm, nodes=eval_nodes, verbose=False
    )

    train_dataset.save_json("train_dataset.json") # TODO: pass in from argument or config file 
    eval_dataset.save_json("eval_dataset.json") # TODO: pass in from argument or config file 
    return train_dataset, eval_dataset


def finetune_embedding_model(train_dataset, eval_dataset, model_id):
    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,
        model_id=model_id, 
        model_output_path="ft_embed_model", # TODO: pass in output path from argument or config file
        val_dataset=eval_dataset,
    )
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()
    return embed_model

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

def evaluate_gold_standard_model(eval_dataset):
    ada = OpenAIEmbedding(api_key='sk-') # TODO: get from os env
    ada_val_results = evaluate(eval_dataset, ada)
    df_ada = pd.DataFrame(ada_val_results)
    hit_rate_ada = df_ada["is_hit"].mean()
    return hit_rate_ada 

def evaluate_pretrained_model(eval_dataset, model_id):
    pt_model = f"local:{model_id}" # TODO: get model name from argument 
    pt_model_eval_results = evaluate(eval_dataset, pt_model) 
    df_pt_model = pd.DataFrame(pt_model_eval_results)
    hit_rate_pt = df_pt_model["is_hit"].mean()
    return hit_rate_pt

def evaluate_finetuned_model(eval_dataset, embed_model):
    val_results_finetuned = evaluate(eval_dataset, embed_model)
    df_finetuned = pd.DataFrame(val_results_finetuned)
    hit_rate_finetuned = df_finetuned["is_hit"].mean()
    return hit_rate_finetuned

if __name__ == "__main__":

    TRAIN_FILES = ["./Scaling_Laws_for_Downstream_Task_Performance_of_Large_Language_Models.pdf"]
    VAL_FILES = ["./Unraveling_the_Mystery_of_Scaling_Laws.pdf"] 
    
    # prepare documents
    train_nodes, eval_nodes = prepare_documents(TRAIN_FILES, VAL_FILES)

    # generate synthetic data
    train_dataset, eval_dataset = generate_qa_pairs(train_nodes, eval_nodes)
    # train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
    # eval_dataset = EmbeddingQAFinetuneDataset.from_json("eval_dataset.json")

    # finetune model
    model_ids = [
        "BAAI/bge-small-en",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    ft_embed_model = finetune_embedding_model(train_dataset, eval_dataset, model_ids[4])

    # evaluate model
    hit_rate_ada = evaluate_gold_standard_model(eval_dataset)
    hit_rate_bge = evaluate_pretrained_model(eval_dataset, model_ids[4])
    hit_rate_ft = evaluate_finetuned_model(eval_dataset, ft_embed_model)
    print(f"Hit rate for Ada: {hit_rate_ada}")
    print(f"Hit rate for BGE: {hit_rate_bge}")
    print(f"Hit rate for Finetuned: {hit_rate_ft}")


'''
BAAI/bge-large-en-v1.5 	1024 	512 	English model
BAAI/bge-base-en-v1.5 	768 	512 	English model
BAAI/bge-small-en-v1.5 	384 	512 	English model

sentence-transformers/all-mpnet-base-v2


'''