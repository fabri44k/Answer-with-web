from duckduckgo_search import DDGS
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import re
from strip_markdown import strip_markdown
from googlesearch import search # https://pypi.org/project/googlesearch-python/
from bs4 import BeautifulSoup
from .duck import DuckDuckGoScraper

class WebScraper:
    
    # [language] -> ddg_region
    # https://pypi.org/project/duckduckgo-search/#regions
    __ddg_regions_mappings = {
        "global": "wt-wt",
        "italian": "it-it",
        "english": "en-us",
        "spanish": "es-es",
        "french": "fr-fr",
        "german": "de-de"
    }
    # [language] -> google_region
    # https://developers.google.com/custom-search/docs/json_api_reference?hl=it#countryCodes
    __google_regions_mappings = {
        "global": "",
        "italian": "it",
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de"
    }

    __SUPPORTED_SEARCH_ENGINES = ["ddg", "google", "ddg_custom"]



    def get_web_links_ddg(self, query, max_results, language):
        region = self.__get_ddg_region_from_language(language)
        results = DDGS().text(query, max_results=max_results, region=region, safesearch='off')
        
        links = [result['href'] for result in results] if results else []
        # fallback to google if no results found
        if not links:
            print("[WARNING] No results found on DDG. Falling back to Google.")
            links = self.get_web_links_google(query, max_results, language)
        
        return links

    def get_web_links_google(self, query, max_results, language):
        region = self.__get_google_region_from_language(language)
        results = search(query, num_results=max_results, region=region, safe=None)
        
        links = [result for result in results if result.startswith("http")] if results else []
        # fallback a google if no results found
        if not links:
            print("[WARNING] No results found on Google. Falling back to DDG.")
            links = self.get_web_links_ddg(query, max_results, language)
        
        return links

    # use my ddg custom scarper
    def get_web_links_ddg_custom(self, query, max_results, language):
        region = self.__get_ddg_region_from_language(language)
        ddg_scraper = DuckDuckGoScraper()
        
        links = ddg_scraper.get_web_links_ddg(query, max_results, region)
        
        # fallback to google if no results found
        if not links:
            print("[WARNING] No results found on DDG Custom. Falling back to Google.")
            links = self.get_web_links_google(query, max_results, language)
        
        return links



    def __get_ddg_region_from_language(self, language):
        
        if language is None or language == "":
            print("[WARNING] Language not specified. Using 'global'.")
            return self.__ddg_regions_mappings["global"]
        
        language = language.lower()
        
        if language in self.__ddg_regions_mappings:
            return self.__ddg_regions_mappings[language]
        else:
            print(f"[WARNING] Language '{language}' not recognized. Using 'global'.")
            return self.__ddg_regions_mappings["global"]


    def __get_google_region_from_language(self, language):
        if language is None or language == "":
            print("[WARNING] Language not specified. Using 'global'.")
            return self.__google_regions_mappings["global"]
        
        language = language.lower()
        
        if language in self.__google_regions_mappings:
            return self.__google_regions_mappings[language]
        else:
            print(f"[WARNING] Language '{language}' not recognized. Using 'global'.")
            return self.__google_regions_mappings["global"]
    


    def print_ddg_supported_languages(self):
        print("Supported languages for DDG:")
        for lang, region in self.__ddg_regions_mappings.items():
            print(f"{lang}: {region}")
    
    def print_google_supported_languages(self):
        print("Supported languages for Google:")
        for lang, region in self.__google_regions_mappings.items():
            print(f"{lang}: {region}")
    
    
    async def __get_content_from_page(self, url, markdown):
        browser_cfg = BrowserConfig(
            headless=True,
            user_agent_mode="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            text_mode=True,
            light_mode=True,
            verbose=False
        )

        crawler_cfg = CrawlerRunConfig(
            scan_full_page=True,
            cache_mode=CacheMode.BYPASS,
            remove_overlay_elements=True,
            exclude_social_media_links=True,
            wait_until="load"
        )
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await crawler.arun(
                url=url,
                config=crawler_cfg
            )
        if markdown:
            return result.markdown
        else:
            return result.html

    @staticmethod
    def __clean_md_before_heading(markdown_content):
        lines = markdown_content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                return '\n'.join(lines[i:])
        return ""

    @staticmethod
    def __remove_markdown_formatting(text):
        text = strip_markdown(text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\{.*?"cl-consent-settings.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'<(style|script)[^>]*>.*?</\1>', '', text, flags=re.DOTALL)
        return text.strip()
    

    def scrape_single_page(self, url, markdown):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        content = loop.run_until_complete(self.__get_content_from_page(url, markdown))

        if not content:
            print("[ERROR] No content found on the page.")
            return
        
        if markdown:
            cleaned_content = self.__clean_md_before_heading(content)
            cleaned_content = self.__remove_markdown_formatting(cleaned_content)
        else:
            # html content
            cleaned_content = content

        if not cleaned_content:
            print("[ERROR] No valid content found after cleaning.")
            return

        return cleaned_content

    def scrape_wikipedia_single_page(self, url):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        content = loop.run_until_complete(self.__get_content_from_page(url, markdown=False))
        
        if not content:
            print("[ERROR] No content found on the page.")
            return
        # get only the paragraphs from the wikipedia page
        cleaned_content = self.__get_paragraphs_from_wikipedia(content)
        
        return cleaned_content


    # knowing that wikipedia is very common and all the text is stored in paragraphs, we can use a specific method to extract only its paragraphs
    def __get_paragraphs_from_wikipedia(self, html):
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = []

        
        blocks = soup.find_all(["p", "dd"])

        for block in blocks:
            # inline math <span>, disabled for now
            # for math_span in block.select("span.mwe-math-element"):
            #     img = math_span.find("img", alt=True)
            #     if img:
            #         alt_text = img["alt"]
            #         math_span.replace_with(alt_text)

            # multiline math in <div>, disabled for now
            # for math_div in block.select("div.mwe-math-element"):
            #     img = math_div.find("img", alt=True)
            #     if img:
            #         alt_text = img["alt"]
            #         math_div.replace_with(alt_text)

            text = block.get_text(strip=False)
            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)



    # find useful links from a query using the specified search engine
    # scrape the content of the pages and return a list of dictionaries with url and content
    def get_scraped_pages(self, query, search_engine, max_pages, language):
        '''Scrape web pages based on a query using the specified search engine.
        Args:

            query (str): The search query.
            search_engine (str): The search engine to use ('ddg' for DuckDuckGo, 'google' for Google).
            max_pages (int): The maximum number of pages to scrape.
            language (str): The language for the search.
        Returns:
            list: A list of dictionaries containing the URL and content of each scraped page.
        '''
        if search_engine not in self.__SUPPORTED_SEARCH_ENGINES:
            print(f"[ERROR] Unsupported search engine: {search_engine}. Supported engines are: {self.__SUPPORTED_SEARCH_ENGINES}")
            return
        
        if search_engine == "ddg":
            links = self.get_web_links_ddg(query, max_results=max_pages, language=language)
        elif search_engine == "google":
            links = self.get_web_links_google(query, max_results=max_pages, language=language)
        elif search_engine == "ddg_custom":
            links = self.get_web_links_ddg_custom(query, max_results=max_pages, language=language)

        else:
            print(f"[ERROR] Unsupported search engine: {search_engine}. Supported engines are: {self.__SUPPORTED_SEARCH_ENGINES}")
            return
        
        if not links:
            print("[ERROR] No link found.")
            return

        pages_data = []
        for url in links:
            
            print(f"\n[OK] Using: {url} ")

            if url.__contains__("wikipedia.org"):
                cleaned_content = self.scrape_wikipedia_single_page(url)
                if not cleaned_content:
                    print(f"[ERROR] Failed to scrape content from {url}.")
                    continue
            else:
                cleaned_content = self.scrape_single_page(url, markdown=True)
                if not cleaned_content:
                    print(f"[ERROR] Failed to scrape content from {url}.")
                    continue


            pages_data.append({
                "url": url,
                "content": cleaned_content
            })
        
        return pages_data
        

