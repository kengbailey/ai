from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
from openai import OpenAI
import requests
from langchain_community.utilities import SearxSearchWrapper
import re
from dataclasses import dataclass



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows the React app to access the API
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class SearchRequest(BaseModel):
    query: str
    num_results: int = 10

@dataclass
class SearchResultSimple:
    Index: int
    Title: str
    Url: str
    Page_Snippet: str 

def replace_urls(text):
    url_pattern_base = r'(https?://\S+)'
    # url_pattern_full = r'\[.*?\]\(.*?\)'
    pass1 = re.sub(url_pattern_base, '[*]', text)
    # pass2 = re.sub(url_pattern_base, '[*]', pass1)
    return pass1

def parse_results_simple(results):
    parsed_results = []
    for i,result in enumerate(results):
        parsed_results.append(SearchResultSimple(Index=i+1,
                                  Title=result['title'],
                                  Url=result['link'],
                                  Page_Snippet=result['snippet']))
    return parsed_results

def call_searxng_search(query, num_results=5, categories=['general']):
    s = SearxSearchWrapper(searx_host="http://192.168.8.116:8877")
    results = s.results(
        query=query, 
        num_results=num_results, 
        categories=categories,
        # time_range="day",
    )
    return results

def call_jina_reader(url, headers=None):
    if not url.startswith("https://r.jina.ai/"):
        url = f"https://r.jina.ai/{url}"
    
    # Set default headers if none are provided
    default_headers = {
        # "x-with-generated-alt": "true", # Enable image captioning
        "Accept": "application/json",
        # "x-respond-with": "text",
    }
    
    if headers:
        default_headers.update(headers) # Add user-provided headers
    
    response = requests.get(url, headers=default_headers)
    
    # Check for successful response
    response.raise_for_status() 

    data = response.text
    
    return json.loads(data)

@app.get("/search-sse")
async def search_sse(request: Request, query: str):
    async def event_generator():
        search_request = await request.json()
        query = search_request['query']
        num_results = search_request.get('num_results', 10)

        # Perform search
        yield {"event": "message", "data": "Performing search..."}
        search_results = call_searxng_search(query, num_results=num_results)
        parsed_results = parse_results_simple(search_results)


        # Generate prompt
        yield {"event": "message", "data": f"Retrieved {len(parsed_results)} results. Feeding to LLM..."}
        prompt_text = f"Query: {request.query}\n\nSearch Results:\n\n"
        for result in parsed_results:
            prompt_text += f"Result: #{result.Index}\nTitle: {result.Title}\nSnippet: {result.Page_Snippet}\n\n"

        # Query LLM
        client = OpenAI(base_url='http://192.168.8.116:11434/v1/', api_key='ollama')
        system_message = """You are an expert assistant at reviewing search results for relevance. Below is a user query and a preliminary list of results. Your job is to give the following information: 
        1. Do the search results contain the answer to the user's query?
        2. What is the answer to the user's query, based on the search results?
        3. Which search results contain the answer to the user's query

        Respond with ONLY a JSON object, looking like this: 

        {
            "has_answer": true,
            "answer": "The answer is 42.",
            "citations": [2,3,5]
        }
        """
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {'role': 'user', 'content': prompt_text}
            ],
            model='llama3.1:8b',
            temperature=0,
        )
        result_json = json.loads(chat_completion.choices[0].message.content)

        # Prepare response
        result_list = []
        for i,result in enumerate(parsed_results):
            result_list.append({
                    'index': i+1,
                    'title': result.Title,
                    'url': result.Url,
                    'page_snippet': result.Page_Snippet,
                    'is_cited': True if (i+1) in result_json['citations'] else False,
                })


        # Prepare and send final response
        ret = {
            'has_answer': result_json['has_answer'],
            'answer': result_json['answer'],
            'citations': result_json['citations'],
            'results': result_list,
        }

        yield {"event": "result", "data": json.dumps(ret)}

    return EventSourceResponse(event_generator())

@app.post("/search")
async def search(request: SearchRequest):
    # Perform search
    search_results = call_searxng_search(request.query, num_results=request.num_results)
    parsed_results = parse_results_simple(search_results)

    # Generate prompt
    prompt_text = f"Query: {request.query}\n\nSearch Results:\n\n"
    for result in parsed_results:
        prompt_text += f"Result: #{result.Index}\nTitle: {result.Title}\nSnippet: {result.Page_Snippet}\n\n"

    # Query LLM
    client = OpenAI(base_url='http://192.168.8.116:11434/v1/', api_key='ollama')
    system_message = """You are an expert assistant at reviewing search results for relevance. Below is a user query and a preliminary list of results. Your job is to give the following information: 
    1. Do the search results contain the answer to the user's query?
    2. What is the answer to the user's query, based on the search results?
    3. Which search results contain the answer to the user's query

    Respond with ONLY a JSON object, looking like this: 

    {
        "has_answer": true,
        "answer": "The answer is 42.",
        "citations": [2,3,5]
    }
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {'role': 'user', 'content': prompt_text}
        ],
        model='llama3.1:8b',
        temperature=0,
    )
    result_json = json.loads(chat_completion.choices[0].message.content)

    # Prepare response
    result_list = []
    for i,result in enumerate(parsed_results):
        result_list.append({
                'index': i+1,
                'title': result.Title,
                'url': result.Url,
                'page_snippet': result.Page_Snippet,
                'is_cited': True if (i+1) in result_json['citations'] else False,
            })

    ret =  {
        'has_answer': result_json['has_answer'],
        'answer': result_json['answer'],
        'citations': result_json['citations'],
        'results': result_list,
    }

    return ret
