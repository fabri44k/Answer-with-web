import os
import re
from dataclasses import dataclass
from typing import Any, Self

from langchain_core.prompts import ChatPromptTemplate

DEFAULT_OLLAMA_URL: str = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class OllamaSettings:
    """Connection settings for the Ollama server, read from the environment / .env."""

    base_url: str = DEFAULT_OLLAMA_URL
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL,
            api_key=os.getenv("OLLAMA_API_KEY") or None,
        )

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def check_ollama_connection(settings: OllamaSettings, model_name: str) -> None:
    """Fail fast if the server is unreachable; warn if the model is not on it."""
    from ollama import Client

    client = Client(host=settings.base_url, headers=settings.headers, timeout=10)
    try:
        available: set[str] = {m.model for m in client.list().models if m.model}
    except Exception as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {settings.base_url}: {e}. "
            "Check OLLAMA_BASE_URL / OLLAMA_API_KEY in your .env file."
        ) from e

    if model_name not in available and f"{model_name}:latest" not in available:
        print(
            f"[WARNING] Model '{model_name}' not found on {settings.base_url}. "
            f"Available: {', '.join(sorted(available)) or 'none'}"
        )


class LLMManager:
    __SUPPORTED_PROVIDERS: list[str] = ["ollama"]

    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float,
        template: str,
        thinking_enabled: bool,
        ollama_settings: OllamaSettings | None = None,
    ) -> None:
        if provider not in self.__SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers are: {
                    self.__SUPPORTED_PROVIDERS
                }"
            )

        self.provider: str = provider
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.thinking_enabled: bool = thinking_enabled
        self.ollama_settings: OllamaSettings = ollama_settings or OllamaSettings()
        self.chain = self.__initialize_llm(template)

    def __initialize_llm(self, template: str) -> Any:
        if not template:
            raise ValueError(
                "Template cannot be empty. Please provide a valid template."
            )

        if self.provider == "ollama":
            from langchain_ollama import OllamaLLM

            model = OllamaLLM(
                model=self.model_name,
                temperature=self.temperature,
                base_url=self.ollama_settings.base_url,
                client_kwargs={"headers": self.ollama_settings.headers},
            )
        else:
            raise Exception("Unable the initialize the model.")

        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        return chain

    def __remove_thinking_from_text(self, text: str) -> str:
        """
        Removes <think>...</think> from response text
        """

        if not text:
            return text
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def answer_query(self, dict_for_template: dict[str, str]) -> str:
        """
        Using a dict containing the required field for the template, invoke the llm and
        return its response
        """
        try:
            response: str = self.chain.invoke(dict_for_template)
        except Exception as e:
            raise RuntimeError(
                f"Error invoking the LLM chain: {
                    e
                }. Check the provider and the template are correctly formatted."
            )

        if self.thinking_enabled:
            return self.__remove_thinking_from_text(response)

        return response.strip()
