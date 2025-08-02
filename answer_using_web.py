import argparse
import json
import os
import re 

from web.web_scraper import WebScraper
from retrieve.st_retrieval import SentenceTransformerRetriever
# from retrieve.bm25_retrieval import BM25Retriever
from llm.llm_manager import LLMManager

CONFIG_FILE = "config.json"
SUPPORTED_SEARCH_ENGINES = ["ddg", "google", "ddg_custom"]
SUPPORTED_RETRIEVAL_MODES = ["sentence_transformers"]
SUPPORTED_LLM_PROVIDERS = ["ollama"]
CSV_SEPARATOR = ";"

def parse_config_json(config_path):
    if not config_path or not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}.")
        exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def init_components(config_path):
    config = parse_config_json(config_path)

    required_keys = ["llm_provider", "final_answer_model", "retrieval_mode", "search_engine", "max_pages", "max_chunk", "llm_template"]
    for key in required_keys:
        if not config.get(key):
            print(f"[ERROR] Missing config key: {key}")
            exit(1)

    if config["llm_provider"] not in SUPPORTED_LLM_PROVIDERS:
        print(f"[ERROR] Unsupported LLM provider: {config['llm_provider']}")
        exit(1)

    if config["search_engine"] not in SUPPORTED_SEARCH_ENGINES:
        print(f"[ERROR] Unsupported search engine: {config['search_engine']}")
        exit(1)

    for m in config["all_llm_configs"]:
        if m["name"] == config["final_answer_model"]:
            temperature = m.get("temperature", 0.3)
            has_thinking = m.get("thinking_enabled", False)
            break
    else:
        print(f"[ERROR] Model config not found: {config['final_answer_model']}")
        exit(1)

    if config["retrieval_mode"] == "sentence_transformers":
        retrieval = SentenceTransformerRetriever(config["embedding_model"])
    else:
        print(f"[ERROR] Unsupported retrieval mode: {config['retrieval_mode']}")
        exit(1)

    return {
        "config": config,
        "retrieval": retrieval,
        "temperature": temperature,
        "has_thinking": has_thinking
    }

def execute_answer_using_web(query, max_pages, language, search_engine, retriever,
                              llm_provider, model_name, max_chunk, save_content_to_file,
                              llm_template, temperature, has_thinking):

    status = "OK" # or NO_WEB_CONTENT or NO_RELEVANT_CHUNKS

    scraper = WebScraper()
    print(f"[INFO] Scraping {max_pages} pages for query: '{query}' in '{language}'")
    data = scraper.get_scraped_pages(query, search_engine=search_engine, max_pages=max_pages, language=language)

    if not data or all("No content found." in page["content"] for page in data):
        status = "NO_WEB_CONTENT"
        scraped_md = ""
    else:
        scraped_md = "\n\n".join([page["content"] for page in data])


    if save_content_to_file:
        with open(f"{query}_scraped_content.md", "w", encoding="utf-8") as f:
            f.write(scraped_md)

    print("[INFO] Finding relevant paragraphs...")
    
    if scraped_md:
        relevant_chunks = retriever.get_relevant_chunks(scraped_md, query, max_chunk)
    else:
        relevant_chunks = [""]

    if (not relevant_chunks or len(relevant_chunks) == 0) and scraped_md:
        print("[WARNING] No relevant chunks found. LLM will answer using the question only.")
        status = "NO_RELEVANT_CHUNKS"
        relevant_chunks = [""]

    if save_content_to_file:
        with open(f"{query}_relevant_content.md", "w", encoding="utf-8") as f:
            f.write("\n\n".join(relevant_chunks))

    dict_for_template = {
        "language": language,
        "question": query,
        "document": "\n\n".join(relevant_chunks)
    }

    llm_manager = LLMManager(llm_provider, model_name, temperature, llm_template, has_thinking)
    
    print(f"[INFO] Generating answer with model: {model_name}...")
    
    final_answer = llm_manager.answer_query(dict_for_template)

    return final_answer, status



def answer_using_web(config_path, query, language, list_languages):
    scraper = WebScraper()

    if list_languages:
        scraper.print_ddg_supported_languages()
        scraper.print_google_supported_languages()
        return "List of supported languages printed."

    init = init_components(config_path)
    cfg = init["config"]

    final_answer, status = execute_answer_using_web(
        query=query,
        max_pages=cfg["max_pages"],
        language=language,
        search_engine=cfg["search_engine"],
        retriever=init["retrieval"],
        llm_provider=cfg["llm_provider"],
        model_name=cfg["final_answer_model"],
        max_chunk=cfg["max_chunk"],
        save_content_to_file=cfg.get("save_content_to_file", False),
        llm_template=cfg["llm_template"],
        temperature=init["temperature"],
        has_thinking=init["has_thinking"]
    )
    
    return {
        "final_answer": final_answer,
        "status": status,
    }



def handle_batch_mode():
    print("[INFO] Batch mode enabled.")

    language = input("[ASK] Language for research (default: global): ").strip() or "global"
    input_file = input("[ASK] Path to input .txt file: ").strip()

    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    expand = input("[ASK] Use query expansion? (y/n): ").strip().lower() == 'y'
    template = input("[ASK] Expansion template, example <Definition of> query, <Synonym of> query: ") if expand else ""

    init = init_components(CONFIG_FILE)
    cfg = init["config"]

    output_file = input_file.replace(".txt", "_answers.csv")
    
    # list of query with warnings
    warning_list = []
    
    with open(output_file, "w", encoding="utf-8") as f:
        for q in questions:
            query = f"{template} {q}" if expand else q
            print(f"[INFO] Processing: {query}")
            
            answer, status = execute_answer_using_web(
                query=query,
                max_pages=cfg["max_pages"],
                language=language,
                search_engine=cfg["search_engine"],
                retriever=init["retrieval"],
                llm_provider=cfg["llm_provider"],
                model_name=cfg["final_answer_model"],
                max_chunk=cfg["max_chunk"],
                save_content_to_file=cfg.get("save_content_to_file", False),
                llm_template=cfg["llm_template"],
                temperature=init["temperature"],
                has_thinking=init["has_thinking"]
            )
            answer = re.sub(CSV_SEPARATOR, ' -', answer)  # Replace CSV separator in answer to avoid issues
            if status != "OK":
                warning_list.append(f"[WARNING] {query} - Status: {status}")
                print(f"[WARNING] {query} - Status: {status}")
            else:
                print(f"[OK] {query} - Answer: {answer}")
            
            f.write(f"{query}" + CSV_SEPARATOR + f"{answer}\n")
    
    if warning_list:
        print("\n[WARNING] Some queries had issues:")
        print("\n".join(warning_list))
    
    
    print(f"[OK] Answers saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer questions using web scraping + LLM.")
    parser.add_argument("-l", type=str, default="global", help="Language for the query.")
    parser.add_argument("-q", type=str, help="Question to ask.")
    parser.add_argument("--list-language", action="store_true", help="List supported languages.")
    parser.add_argument("-b", "--batch", action="store_true", help="Enable batch mode.")
    args = parser.parse_args()

    if args.batch:
        handle_batch_mode()
    elif not args.q and not args.list_language:
        print("[ERROR] Please provide a question with -q")
        exit(1)
    else:
        answer = answer_using_web(CONFIG_FILE, args.q, args.l, args.list_language)
        if not args.list_language:
            print(f"\n[OK] Final answer with status {answer['status']}:\n")
            print(answer["final_answer"])
            print("\n[INFO] Done.")
