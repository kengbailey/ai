from openai import OpenAI
from ollama import Client
import wikipedia
import lancedb

from nltk.tokenize import sent_tokenize
import nltk
import re



nltk.download("punkt")


"""
Things we want to do

1. Submit text to a large model and ask questions we can ask of the text
- the goal is to use a large model to find needles in the hackstack 
- then we can use a lesser model to answer questions about the hackstack
- this will be nice for testing the embedding models in ollama

2. We want to test the various embedding models in ollama 
- we have dimensions from 384 to 1024
- keeping everything the same, we want to see how well the various models handle the questions

3. We want to have some structure to how we are recording this information
- we may want to use a database, but that may impede the speed at which we work. 
- we'll create the structure and then we can use the after we've identified all the variables to capture

4. We want to 



Notes
- store the various models in separate tables 
- then we need to change the model to a new one each time
- let's see exactly how this will change over



"""


# Function to prepare data for the database
def prepare_data(chunks, embeddings):
    data = []
    for chunk, embed in zip(chunks, embeddings):
        temp = {}
        temp["text"] = chunk
        temp["vector"] = embed
        data.append(temp)
    return data

# Function to create a table in lancedb, with the model name and dimensions
def create_lancedb_table(chunks, embeddings, data_name, model_name):
    table_name = f"{data_name}_{model_name}"  
    db = lancedb.connect("lance.db")
    data = prepare_data(chunks, embeddings)
    table = db.create_table(
        table_name,
        data=data,
        mode="overwrite",
    )
    return table

# Function to do similarity search on embeds in lancedb table
# Retriever
def search_lancedb(table, question, query_k=5):
    query_embedding = get_embedding(question)
    result = table.search(query_embedding).limit(query_k).to_list()
    return [r["text"] for r in result] 

# Function to create embeddings from data ingested
def get_embedding(text, model='all-minilm:22m'):
    client = Client(host='http://192.168.8.116:11434')
    response = client.embeddings(
        prompt=text,
        model=model
    )
    return response["embedding"]
    

# Function to ingest data
def ingest_data_wikipedia(article_name):
    page = wikipedia.page(article_name)
    with open(f'{article_name}.txt', 'w') as f:
        f.write(page.content)
    print(f"Article successfully written article( {article_name} ) to text")
    return page.content


def recursive_text_splitter(text, max_chunk_length=1000, overlap=100):
    # Initialize result
    result = []

    current_chunk_count = 0
    separator = ["\n", " "]
    _splits = re.split(f"({separator})", text)
    splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

    for i in range(len(splits)):
        if current_chunk_count != 0:
            chunk = "".join(
                splits[
                    current_chunk_count
                    - overlap : current_chunk_count
                    + max_chunk_length
                ]
            )
        else:
            chunk = "".join(splits[0:max_chunk_length])

        if len(chunk) > 0:
            result.append("".join(chunk))
        current_chunk_count += max_chunk_length

    return result



def create_prompt(question, context):
    base_instruction = """Your task is to understand the user question, and provide an answer using the provided contexts. Every answer you generate should have citations in this pattern  "Answer [position].", for example: "Earth is round. [1][2]," if it's relevant.

    Your answers are correct, high-quality, and written by an domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."
    \n\n"""
    question_text = "User Question: {}\n\nContexts:\n"
    context_text = "[{}]\n{}\n\n"

    prompt = ''
    for i,text in enumerate(context):
        prompt += f"{context_text.format(i+1, text)}"
    prompt = base_instruction + f"{question_text.format(question)}" + prompt
    
    return prompt


def query_llm(prompt, model):

    client = OpenAI(
        base_url='http://192.168.8.116:11434/v1/',
        api_key='ollama',
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {'role': 'user', 'content': prompt, }
        ],
        model=model,
        temperature=0,
    )
    return chat_completion.choices[0].message.content


if __name__ == "__main__":

    #### SETUP
    ##########
    article_name = ""
    chunk_length = 1000
    chunk_overlap = 100

    # Ingest data
    text = ingest_data_wikipedia(article_name)

    # Chunk data // split text
    chunks = recursive_text_splitter(text, chunk_length, chunk_overlap)

    # vectorize data // create embeddings
    embeds = []
    for chunk in chunks:
        embed = get_embedding(chunk)
        embeds.append(embed)

    # create and insert lancedb table
    embed_table = create_lancedb_table(chunks, embeds, article_name.replace(" ", ""))

    #### Query
    ##########
    question = ""
    model = ""

    # query lancedb table 
    results = search_lancedb(embed_table, question)
    
    # build prompt
    prompt = create_prompt(question, results)
    
    # query LLM
    llm_result = query_llm(prompt)

    # log results --> question, answer, model, article_name,  chunk_length, chunk_overlap, embedding_model, prompt


    # fin 