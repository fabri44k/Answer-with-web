import requests
import random
from typing import List, Dict, Union
from bs4 import BeautifulSoup

# use simple request and user agent rotation to scrape DuckDuckGo search results
class DuckDuckGoScraper:

    __DDG_URL = 'https://html.duckduckgo.com/html'
    __MAX_RESULT_FOR_PAGE_DDG = 10
    __USER_AGENT_FILE_PATH = './web/user_agents.txt'  # Path from main file to user agents file


    def __get_random_user_agent(self) -> str:
        with open(self.__USER_AGENT_FILE_PATH, 'r') as file:
            user_agents = file.readlines()
            return random.choice(user_agents).strip()

    def __get_ddg_html_content(self, query: str, region: str = 'wt-wt') -> str:
        user_agent = self.__get_random_user_agent()
        headers = {
            'User-Agent': user_agent
        }

        params = {
            'q': query,
            'kl': region,
        }

        response = requests.post(self.__DDG_URL, headers=headers, params=params)

        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to retrieve content: {response.status_code}")

    def __parse_ddg_result_page(self, html: Union[str, bytes], max_results: int) -> List[Dict[str, str]]:
        '''
        Parses the HTML content of a DuckDuckGo search results page and extracts the links.
        Returns a list of dictionaries with 'title', 'description', and 'url'.
        '''
        
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for i in soup.find_all('div', {'class': 'links_main'}):
            if i.find('a', {'class': 'badge--ad'}):
                continue
            try:
                title = i.h2.a.text
                description = i.find('a', {'class': 'result__snippet'}).text
                url = i.find('a', {'class': 'result__url'}).get('href')
                
                if not title or not url:
                    continue
                
               
                results.append({
                    'title': title,
                    'description': description,
                    'url': url
                })

                if len(results) >= max_results:
                    break
            
            except AttributeError:
                pass
        
        
        return results

    def get_web_links_ddg(self, query, max_results, region):

        if max_results <= 0:
            raise ValueError("max_results must be greater than 0")

        if max_results > self.__MAX_RESULT_FOR_PAGE_DDG:
            raise ValueError(f"max_results for ddg must be less than or equal to {self.__MAX_RESULT_FOR_PAGE_DDG}")

        html_content = self.__get_ddg_html_content(query, region)
        results = self.__parse_ddg_result_page(html_content, max_results)

        # return only the first max_results urls
        return [result['url'] for result in results[:max_results]]
