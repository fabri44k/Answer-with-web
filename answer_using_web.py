import argparse
import json
import os
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv

from llm.llm_manager import LLMManager, OllamaSettings, check_ollama_connection

# WebScraper (crawl4ai) and SentenceTransformerRetriever (torch) are imported lazily,
# so that --no-web mode starts fast and does not load the embedding model.

CONFIG_FILE: str = "config.json"
ENV_FILE: Path = Path(__file__).parent / ".env"
SUPPORTED_SEARCH_ENGINES: list[str] = ["ddg", "google", "ddg_custom"]
SUPPORTED_RETRIEVAL_MODES: list[str] = ["sentence_transformers"]
SUPPORTED_LLM_PROVIDERS: list[str] = ["ollama"]
CSV_SEPARATOR: str = ";"


def parse_config_json(config_path):
    if not config_path or not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}.")
        exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_config_yaml(file):
    if file is None:
        return 0


def init_components(config_path, use_web=True):
    config = parse_config_json(config_path)

    required_keys = ["llm_provider", "final_answer_model", "llm_template"]
    if use_web:
        required_keys += ["retrieval_mode", "search_engine", "max_pages", "max_chunk"]

    for key in required_keys:
        if not config.get(key):
            print(f"[ERROR] Missing config key: {key}")
            exit(1)

    if config["llm_provider"] not in SUPPORTED_LLM_PROVIDERS:
        print(f"[ERROR] Unsupported LLM provider: {config['llm_provider']}")
        exit(1)

    if use_web and config["search_engine"] not in SUPPORTED_SEARCH_ENGINES:
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

    # Connection settings come from the environment / .env, not from config.json.
    # Variables already set in the shell take precedence over the .env file.
    load_dotenv(ENV_FILE)
    ollama_settings = OllamaSettings.from_env()
    try:
        check_ollama_connection(ollama_settings, config["final_answer_model"])
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        exit(1)
    print(f"[INFO] Using Ollama at {ollama_settings.base_url}")

    retrieval = None
    if use_web:
        if config["retrieval_mode"] == "sentence_transformers":
            from retrieve.st_retrieval import SentenceTransformerRetriever

            retrieval = SentenceTransformerRetriever(config["embedding_model"])
        else:
            print(f"[ERROR] Unsupported retrieval mode: {config['retrieval_mode']}")
            exit(1)
        llm_template = config["llm_template"]
    else:
        # Fall back to the web template (with an empty document) if no dedicated one
        llm_template = config.get("llm_template_no_web") or config["llm_template"]

    return {
        "config": config,
        "retrieval": retrieval,
        "llm_template": llm_template,
        "temperature": temperature,
        "has_thinking": has_thinking,
        "ollama_settings": ollama_settings,
    }


def execute_answer_using_web(
    query,
    max_pages,
    language,
    search_engine,
    retriever,
    llm_provider,
    model_name,
    max_chunk,
    save_content_to_file,
    llm_template,
    temperature,
    has_thinking,
    use_web=True,
    ollama_settings: OllamaSettings | None = None,
):
    if not use_web:
        llm_manager = LLMManager(
            llm_provider,
            model_name,
            temperature,
            llm_template,
            has_thinking,
            ollama_settings,
        )
        print(f"[INFO] Web disabled, generating answer with model: {model_name}...")
        final_answer = llm_manager.answer_query(
            {"language": language, "question": query, "document": ""}
        )
        return final_answer, "WEB_DISABLED"

    from web.web_scraper import WebScraper

    status = "OK"  # or NO_WEB_CONTENT or NO_RELEVANT_CHUNKS

    scraper = WebScraper()
    print(f"[INFO] Scraping {max_pages} pages for query: '{query}' in '{language}'")
    data = scraper.get_scraped_pages(
        query, search_engine=search_engine, max_pages=max_pages, language=language
    )

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
        print(
            "[WARNING] No relevant chunks found. LLM will answer using the question only."
        )
        status = "NO_RELEVANT_CHUNKS"
        relevant_chunks = [""]

    if save_content_to_file:
        with open(f"{query}_relevant_content.md", "w", encoding="utf-8") as f:
            f.write("\n\n".join(relevant_chunks))

    dict_for_template = {
        "language": language,
        "question": query,
        "document": "\n\n".join(relevant_chunks),
    }

    llm_manager = LLMManager(
        llm_provider,
        model_name,
        temperature,
        llm_template,
        has_thinking,
        ollama_settings,
    )

    print(f"[INFO] Generating answer with model: {model_name}...")

    final_answer = llm_manager.answer_query(dict_for_template)

    return final_answer, status


def answer_using_web(config_path, query, language, list_languages, use_web=True):
    if list_languages:
        from web.web_scraper import WebScraper

        scraper = WebScraper()
        scraper.print_ddg_supported_languages()
        scraper.print_google_supported_languages()
        return "List of supported languages printed."

    init = init_components(config_path, use_web)
    cfg = init["config"]

    final_answer, status = execute_answer_using_web(
        query=query,
        max_pages=cfg.get("max_pages"),
        language=language,
        search_engine=cfg.get("search_engine"),
        retriever=init["retrieval"],
        llm_provider=cfg["llm_provider"],
        model_name=cfg["final_answer_model"],
        max_chunk=cfg.get("max_chunk"),
        save_content_to_file=cfg.get("save_content_to_file", False),
        llm_template=init["llm_template"],
        temperature=init["temperature"],
        has_thinking=init["has_thinking"],
        ollama_settings=init["ollama_settings"],
        use_web=use_web,
    )

    return {
        "final_answer": final_answer,
        "status": status,
    }


def batch_output_path(input_file: Path) -> Path:
    """`terms.txt` -> `terms_answers.csv`, next to the input file."""
    return input_file.with_name(f"{input_file.stem}_answers.csv")


def sanitize_csv_field(text: str) -> str:
    """Keep a value on a single CSV field: no separator, no line breaks."""
    return " ".join(text.replace(CSV_SEPARATOR, " -").split())


def drop_incomplete_last_line(path: Path) -> None:
    """Remove a truncated last row left by a crash in the middle of a write."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


def load_answered_queries(path: Path) -> set[str]:
    """Queries already saved in the output CSV, used to resume an interrupted batch."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {line.split(CSV_SEPARATOR, 1)[0] for line in f if CSV_SEPARATOR in line}


def append_row(f: TextIO, query: str, answer: str) -> None:
    """Write one row and force it to disk, so a crash cannot lose it."""
    f.write(f"{query}{CSV_SEPARATOR}{answer}\n")
    f.flush()
    os.fsync(f.fileno())


def handle_batch_mode(use_web: bool = True) -> None:
    print("[INFO] Batch mode enabled.")
    if not use_web:
        print("[INFO] Web disabled: the LLM will answer from its own knowledge.")

    language = (
        input("[ASK] Language for research (default: global): ").strip() or "global"
    )
    input_file = Path(input("[ASK] Path to input .txt file: ").strip())

    if not input_file.is_file():
        print(f"[ERROR] File not found: {input_file}")
        exit(1)

    with input_file.open(encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    expand = input("[ASK] Use query expansion? (y/n): ").strip().lower() == "y"
    template = (
        input(
            "[ASK] Expansion template, example <Definition of> query, <Synonym of> query: "
        )
        if expand
        else ""
    )

    init = init_components(CONFIG_FILE, use_web)
    cfg = init["config"]

    output_file = batch_output_path(input_file)
    drop_incomplete_last_line(output_file)
    answered = load_answered_queries(output_file)

    # (query sent to the pipeline, sanitized query written in the CSV)
    all_queries: list[tuple[str, str]] = []
    for q in questions:
        query = f"{template} {q}" if expand else q
        all_queries.append((query, sanitize_csv_field(query)))
    pending = [(query, key) for query, key in all_queries if key not in answered]

    if skipped := len(all_queries) - len(pending):
        print(f"[INFO] {skipped} queries already answered in {output_file}, skipping.")

    warning_list: list[str] = []  # saved, but answered without web content
    error_list: list[str] = []  # not saved, will be retried on the next run
    saved = 0

    try:
        with output_file.open("a", encoding="utf-8") as f:
            for i, (query, key) in enumerate(pending, start=1):
                print(f"[INFO] ({i}/{len(pending)}) Processing: {query}")

                try:
                    answer, status = execute_answer_using_web(
                        query=query,
                        max_pages=cfg.get("max_pages"),
                        language=language,
                        search_engine=cfg.get("search_engine"),
                        retriever=init["retrieval"],
                        llm_provider=cfg["llm_provider"],
                        model_name=cfg["final_answer_model"],
                        max_chunk=cfg.get("max_chunk"),
                        save_content_to_file=cfg.get("save_content_to_file", False),
                        llm_template=init["llm_template"],
                        temperature=init["temperature"],
                        has_thinking=init["has_thinking"],
                        ollama_settings=init["ollama_settings"],
                        use_web=use_web,
                    )
                except Exception as e:
                    error_list.append(f"[ERROR] {query} - {e}")
                    print(f"[ERROR] {query} - {e}")
                    continue

                answer = sanitize_csv_field(answer)
                if not answer:
                    error_list.append(f"[ERROR] {query} - Empty answer")
                    print(f"[ERROR] {query} - Empty answer")
                    continue

                append_row(f, key, answer)
                saved += 1

                if status not in ("OK", "WEB_DISABLED"):
                    warning_list.append(f"[WARNING] {query} - Status: {status}")
                    print(f"[WARNING] {query} - Status: {status}")
                else:
                    print(f"[OK] {query} - Answer: {answer}")
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted by user.")

    if warning_list:
        print("\n[WARNING] Some queries had issues:")
        print("\n".join(warning_list))

    if error_list:
        print("\n[ERROR] Some queries failed and were not saved:")
        print("\n".join(error_list))

    print(f"\n[OK] {saved} new answers saved to {output_file}")
    if missing := len(pending) - saved:
        print(f"[INFO] {missing} queries missing: run again to retry only those.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Answer questions using web scraping + LLM."
    )
    parser.add_argument(
        "-l", type=str, default="global", help="Language for the query."
    )
    parser.add_argument("-q", type=str, help="Question to ask.")
    parser.add_argument(
        "--list-language", action="store_true", help="List supported languages."
    )
    parser.add_argument("-b", "--batch", action="store_true", help="Enable batch mode.")
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Skip web fetch and retrieval: the LLM answers from its own knowledge.",
    )
    args = parser.parse_args()
    use_web = not args.no_web

    if args.batch:
        handle_batch_mode(use_web)
    elif not args.q and not args.list_language:
        print("[ERROR] Please provide a question with -q")
        exit(1)
    else:
        answer = answer_using_web(
            CONFIG_FILE, args.q, args.l, args.list_language, use_web
        )
        if not args.list_language:
            print(f"\n[OK] Final answer with status {answer['status']}:\n")
            print(answer["final_answer"])
            print("\n[INFO] Done.")
