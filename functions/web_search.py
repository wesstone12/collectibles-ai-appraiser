from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
import json
from utils.retry import retry_with_backoff
import time
from loguru import logger

@retry_with_backoff(max_retries=3, initial_delay=2, max_delay=15)
def text_search(query: str, num_results: int = 3) -> str:
    """Conducts a web search for Pokemon card prices and market data.
    
    :param query: The search query string for finding relevant price information.
    :param num_results: The maximum number of URLs to return. Defaults to 3.
    :return: A JSON-formatted string containing the search results.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"{query} pokemon card price tcgplayer ebay",
                max_results=num_results
            ))
            
            processed_results = []
            for result in results:
                try:
                    response = requests.get(result['href'], timeout=10)
                    if response.status_code == 429:  # Rate limit hit
                        raise Exception("Rate limit reached")
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = ' '.join(soup.stripped_strings)
                    text = ' '.join(text.split())[:1000]  # Limit text length
                    processed_results.append({
                        'title': result['title'],
                        'url': result['href'],
                        'content': text
                    })
                except requests.exceptions.RequestException:
                    continue
                    
            if not processed_results:
                raise Exception("No results could be processed")
                
            return json.dumps(processed_results)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise 