# Answer with Web

Answer with Web gives you a short LLM answer for one query, or for a long list of them. You can also feed the LLM relevant content scraped from the web.
Its main use today is building **Anki decks** in batch: you give it a list of questions and get a ready-to-import CSV of `question;answer` pairs.

Here is what it does:

- (OPTIONAL) WEB_FETCH: Scrapes web pages based on a user query using a search engine (Google or DuckDuckGo).
- (IF WEB_FETCH) RETRIEVAL: Extracts relevant content from the scraped pages.
- ANSWER: Uses a large language model (LLM) to generate a final answer for your query, based on the extracted content when present.

Example use case: you have a list of 100 words (e.g. `Meaning of perorare`, `Meaning of ...`) and want a concise definition for each one, all in the same format, as flashcards.

### When does it make sense?

Today agentic tools (Claude Code, ChatGPT, local models with tool calling) can search the web on their own, so this project is no longer the only way to get grounded answers. It still makes sense when:

- **You have a large batch of similar queries.** A fixed pipeline beats an agent here: the cost is predictable, every answer follows the same template (1-3 sentences, ideal for flashcards), results are reproducible, and you don't need to babysit 100 agent runs.
- **You don't want to pay anything.** Everything runs locally with Ollama. No subscription, no API keys, no rate limits.
- **You use small local models.** They hallucinate more often, and grounding them on retrieved web content makes their answers noticeably more reliable.
- **You want to save tokens.** The model gets only a few relevant chunks instead of searching, opening and reading whole pages by itself.
- **You want to learn.** It is a small, readable, end-to-end RAG pipeline: search, scraping, chunking, embeddings, ranking, generation.

### Limitations

- **The setup is heavy.** Torch with CUDA, Playwright, crawl4ai, an embedding model and Ollama are all required, so it is not really suited for very limited hardware.
- **Scraping is fragile.** Search engines change and add bot protection, which can break the fetch step.
- **Quality depends on the top results.** With a low `max_pages` the answer depends on the first pages returned by the search engine.
- **Web content isn't always needed.** For common knowledge (e.g. definitions of everyday words) a capable model already knows the answer, and retrieval adds time without adding quality. In that case use `--no-web`. Keep in mind that small local models tend to invent definitions of less common words, so with them the web fetch is still worth it.


## Examples of Usage

### Single query

![Basic Usage](./docs/img/basic.png)

### Batch Mode

![Batch Usage](./docs/img/batch.png)


#### Output

```csv
Definition of ollama;Ollama is a machine learning platform that democratizes access to large language models (LLMs) by enabling users to run them locally on their machines. It offers a user-friendly interface and seamless integration capabilities for various applications in AI and natural language processing.
Definition of mysql;MySQL is an open-source relational database management system that uses SQL for managing and manipulating data. It is developed by Oracle Corporation and supports multiple platforms like Windows, Linux, and macOS.
Definition of torch python;PyTorch is an open-source deep learning library built on Python, providing GPU acceleration, dynamic computation graphs, and an intuitive interface for researchers and developers. It follows a "define-by-run" approach with flexible model execution and debugging using dynamic graphs.
Definition of type casting in C;Type casting in C is converting one data type to another using the casting operator (e.g., `(int)` or `(float)`. It's used when a variable's datatype needs changing for specific operations. There are two main types: explicit (forced conversion) and implicit (automatic conversion).
```

This file is ready to be imported into Anki.


## Install

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

- Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) installed. The correct Python version (3.13) is downloaded automatically by uv, so you don't need to install Python yourself.

- Requires Ollama installed and running on your machine.

- Install all dependencies (including torch with CUDA 12.6 support):

```bash
uv sync
```

The CUDA build of torch is configured in `pyproject.toml` and pinned in `uv.lock`, so `uv sync` installs the right wheel automatically on both Linux and Windows. If you don't have a GPU the execution will still work but of course it will be slower.

- Setup web scraper

```bash
uv run playwright install
uv run crawl4ai-setup
uv run crawl4ai-doctor
```
Ensure that at the end appears: `[COMPLETE] ● ✅ Crawling test passed!`

- Choose and download an Ollama model from https://ollama.com/search, this will be used to construct the final question.

- The next step is to configure the `config.json` file, see the configuration section below for details.

## Configuration

Here is the explanation of every parameter you can adjust in `config.json`:

- `llm_provider`: The provider of the large language model (LLM) to use for generating responses. Right now only `ollama` is supported.

- `final_answer_model`: The specific LLM model to use from the provider. For Ollama, this must match a model you have installed locally (get their names running `ollama list`). This model must be specified in the `all_llm_configs` section at the bottom as well.

- `retrieval_mode`: The method used for semantic retrieval of relevant content. Supported: `sentence_transformers`.

- `embedding_model`: Name of the embedding model used to convert text into vectors for semantic search. Example: `"Qwen/Qwen3-Embedding-0.6B"`. This model must be available on Hugging Face and will be downloaded automatically if not cached. Select an embedding model from https://huggingface.co/models?library=sentence-transformers, here is a rank for multilingual capabilities: https://huggingface.co/spaces/mteb/leaderboard

- `search_engine`: The web search engine used to retrieve documents. Supported: `google`, `ddg` (DuckDuckGo), `ddg_custom` which is my custom and simpler DuckDuckGo Scraper.

- `max_pages`: Maximum number of web pages to scrape for a given query. A higher value will increase the time required to scrape and process the content.

- `max_chunk`: Maximum number of relevant chunks to extract from the scraped document to be passed to the LLM.

- `save_content_to_file`: If `true`, saves the scraped content and selected chunks to local markdown files for inspection/debugging.

- `llm_template`: The prompt template used to instruct the LLM. It includes placeholders like `{language}`, `{question}`, and `{document}` that are filled at runtime. This guides the model to generate accurate, concise, and language-specific answers.

- `llm_template_no_web` (optional): The prompt template used with `--no-web`. It supports `{language}` and `{question}`. If missing, `llm_template` is used with an empty document.

- `all_llm_configs`: A list of configurations for available LLM models (only ollama is supported). Each object must include:
  - `name`: The name of the model (must match what is used in `final_answer_model`)
  - `temperature`: Sampling temperature for the model (lower is more deterministic)
  - `thinking_enabled`: Set this to true if the model supports thinking steps (like `deepseek-r1`), which allows it to reason before answering, any thinking steps will not be included in the final answer.

### Guidelines

- A good temperature for the final answer model is around `0.2` to `0.4`, which balances creativity and accuracy.
- Consider using a higher `max_chunk` value if you have a model with a large context.
- Choose the best llm depending on the language you want to use, during my test I've found that `mistral-nemo` works well for Italian, `gemma3` performs well in English, but it may not be the best choice for other languages.
- You can modify llm behavior by adjusting the `llm_template`, for example, you can add more instructions to guide the model on how to answer the question.

Once the config file is completed, you can start using tool.

## Features

### Search engine region

It use a parameter to specify the region of the search engine, which can help in retrieving more relevant results based on your location or the language of the query.
This is specified by the `-l` parameter, which should match a language supported by the search engine.

### Search engine fallback

If somehow the search engine fails to retrieve results, the tool will try to use a different search engine to ensure that you still get relevant content. Example if google fails, it will try DuckDuckGo.

### Wikipedia enhanced scraping

Since Wikipedia is a common source of information, the tool has a special scraping method for Wikipedia pages that extracts text content more effectively than the standard web scraping.

### Query expansion

Query expansion is useful in batch mode, in order to save you time. If you have a **list of keywords** or phrases, you can use a query expansion template to generate more specific and informative queries for each item.

You can insert a prefix or full sentence (template) before each query to better contextualize it and improve the relevance of the results. Then it will run the same procedure for each template+keyword generated.

Example: `Meanings of` as a template will execute for each query `Meanings of {query}?`.

Other examples of templates you can use:

- Meanings of {query}?

- Explain in simple terms the concept of {query}

- Provide examples of {query}

- Opposite of {query}?

- Synonyms of {query}


### Response status

The tool provides a status for each response, indicating whether the answer was generated successfully, if it was based on web content, or if it was generated from the LLM without web content. This helps you understand the reliability of the answer.

- `status` can be one of the following:
  - `OK`: The answer was generated successfully.
  - `NO_WEB_CONTENT`: The answer was generated from the LLM without using web content.
  - `NO_RELEVANT_CHUNKS`: The retrieval did not find any relevant chunks from the web content. The answer will be based on the LLM without web content.
  - `WEB_DISABLED`: The web fetch was skipped on purpose with `--no-web`. The answer comes from the LLM's own knowledge.

### No-web mode

With `--no-web` the tool skips the web fetch and the embedding retrieval: the LLM answers directly using `llm_template_no_web`. The scraper and the embedding model are not even loaded, so it starts much faster and uses less memory. It works in both single query and batch mode, and with `search_engine`, `max_pages`, `max_chunk` and `retrieval_mode` not set in the config.


## Usage

### Single Query mode

```bash
uv run python3 answer_using_web.py -l <language> -q "<query>"
```

- `-l` specifies the language used in the search engine and in the output response. It should match a language supported by the search engine.

Get the supported languages for the search engine you are using:

```bash
uv run python3 .\answer_using_web.py --list-language
```

- `-q` is the question you want to ask. It should be in the language specified by `-l`.

- `--no-web` (optional) skips the web fetch and retrieval, and the LLM answers directly.

```bash
uv run python3 answer_using_web.py --no-web -l <language> -q "<query>"
```

## Batch Mode


Activate the batch mode:

```bash
uv run python3 answer_using_web.py -b
```

Add `--no-web` to run the whole batch without the web fetch: `uv run python3 answer_using_web.py -b --no-web`.

It will ask:

- Specifies the language used in the search engine and in the output response. It should match a language supported by the search engine.
- The path to the input .txt file containing questions.
- The query expansion template (optional).

It will show at the end if some question has a status different than `OK`.

The output will be a CSV file with the answers generated (`question; answer`), saved in the same directory as the input file.


## Disclaimer


The possibile mistakes are:

- Web scarping fails to retrieve content due to bot-protection mechanisms or other issues.
- The retrieval process does not find relevant chunks, this is due to the input document and the embedding model.
- The LLM may generate answers that are not fully accurate or relevant, especially if the input question is ambiguous or lacks context.
- The quality is highly dependent on the LLM model used, so choose a model that is well-suited for your language and task. Do some tests with different models to find the best one for your needs.


## History and my use case

I wrote this project in 2025, when local LLMs could not call tools. A model running in Ollama had no way to search the web, so small open-source models often gave confident but wrong answers. The idea was to do the search myself and hand the model the right context, getting something similar to ChatGPT's web search but fully local and free.

It was also an educational project: a way for me to understand, hands-on, how a retrieval-augmented generation (RAG) pipeline works, from scraping to embeddings to prompt design.

Things have changed since then: agentic harnesses and tool-calling local models can now browse the web on their own. I still use this tool for one job: building Anki decks from long lists of terms, where a simple, predictable, free batch pipeline is the most practical option.

