from __future__ import annotations

import argparse
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import requests
from requests import Response
from transformers import pipeline


class SearchEngine(str, Enum):
    BING = "bing"
    GOOGLE = "google"
    DUCKDUCKGO = "duckduckgo"


def _safe_request(method: str, url: str, **kwargs) -> Optional[Response]:
    try:
        response = requests.request(method=method, url=url, timeout=10, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        print(f"Request to {url} failed: {exc}")
        return None


def search_bing(query: str, num_results: int) -> str:
    api_key = os.getenv("BING_API_KEY")
    if not api_key:
        raise EnvironmentError("BING_API_KEY environment variable is not set.")
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": num_results}
    response = _safe_request("GET", endpoint, headers=headers, params=params)
    if response is None:
        return ""
    results: Dict = response.json()
    snippets = [item.get("snippet", "") for item in results.get("webPages", {}).get("value", [])]
    return " ".join(snippets)


def search_google(query: str, num_results: int) -> str:
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise EnvironmentError("SERPAPI_KEY environment variable is not set.")
    endpoint = "https://serpapi.com/search"
    params = {"q": query, "num": num_results, "api_key": api_key, "engine": "google"}
    response = _safe_request("GET", endpoint, params=params)
    if response is None:
        return ""
    results: Dict = response.json()
    snippets = [item.get("snippet", "") for item in results.get("organic_results", [])]
    return " ".join(snippets)


def search_duckduckgo(query: str, num_results: int) -> str:
    endpoint = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
    response = _safe_request("GET", endpoint, params=params)
    if response is None:
        return ""
    results: Dict = response.json()
    snippets = []
    for item in results.get("RelatedTopics", [])[:num_results]:
        text = item.get("Text")
        if text:
            snippets.append(text)
    return " ".join(snippets)


def search_web(query: str, num_results: int, engine: SearchEngine) -> str:
    if engine is SearchEngine.BING:
        return search_bing(query, num_results)
    if engine is SearchEngine.GOOGLE:
        return search_google(query, num_results)
    if engine is SearchEngine.DUCKDUCKGO:
        return search_duckduckgo(query, num_results)
    raise ValueError(f"Unsupported search engine: {engine}")


def call_ollama(prompt: str, model: str) -> str:
    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = _safe_request("POST", endpoint, json=payload)
    if response is None:
        return ""
    result = response.json()
    return result.get("response", "")


def generate_response(
    query: str,
    engine: SearchEngine,
    num_results: int,
    use_ollama: bool,
    ollama_model: str,
) -> str:
    context = search_web(query, num_results=num_results, engine=engine)
    prompt = f"Answer the following question using the context:\nContext: {context}\nQuestion: {query}\nAnswer:"
    if use_ollama:
        return call_ollama(prompt=prompt, model=ollama_model)
    generator = pipeline("text-generation", model="gpt2")
    response = generator(prompt, max_length=200, do_sample=True)
    return response[0]["generated_text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the web and generate an answer using an LLM.")
    parser.add_argument("query", help="Question to ask.")
    parser.add_argument("--engine", choices=[engine.value for engine in SearchEngine], default=SearchEngine.BING.value)
    parser.add_argument("--num-results", type=int, default=3, help="Number of snippets to aggregate.")
    parser.add_argument("--use-ollama", action="store_true", help="Use the local Ollama endpoint instead of GPT-2.")
    parser.add_argument("--ollama-model", default="llama2", help="Model name exposed by Ollama.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = SearchEngine(args.engine)
    answer = generate_response(
        query=args.query,
        engine=engine,
        num_results=args.num_results,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
    )
    print(f"Response:\n{answer}")


if __name__ == "__main__":
    main()
    