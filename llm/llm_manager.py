import os
import re
from dataclasses import dataclass
from typing import Any, Self

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

SUPPORTED_PROVIDERS: tuple[str, ...] = ("ollama", "openai_compatible")
DEFAULT_OLLAMA_URL: str = "http://127.0.0.1:11434"

# Prefix of the environment variables (<PREFIX>_BASE_URL, <PREFIX>_API_KEY) per provider
ENV_PREFIXES: dict[str, str] = {
    "ollama": "OLLAMA",
    "openai_compatible": "OPENAI_COMPATIBLE",
}


@dataclass(frozen=True)
class ProviderSettings:
    """Connection settings for the LLM provider, read from the environment / .env."""

    provider: str
    base_url: str
    api_key: str | None = None

    @classmethod
    def from_env(cls, provider: str) -> Self:
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported: {SUPPORTED_PROVIDERS}"
            )

        prefix = ENV_PREFIXES[provider]
        base_url = os.getenv(f"{prefix}_BASE_URL") or None
        if base_url is None:
            if provider != "ollama":
                raise ValueError(
                    f"{prefix}_BASE_URL is not set. Add it to your .env file "
                    "(see .env.example)."
                )
            base_url = DEFAULT_OLLAMA_URL

        return cls(
            provider=provider,
            base_url=base_url.rstrip("/"),
            api_key=os.getenv(f"{prefix}_API_KEY") or None,
        )

    @property
    def env_prefix(self) -> str:
        return ENV_PREFIXES[self.provider]

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def model_is_available(model_name: str, available: set[str]) -> bool:
    # Ollama lists untagged models as "<name>:latest"
    return model_name in available or f"{model_name}:latest" in available


def list_models(settings: ProviderSettings) -> set[str]:
    """Model ids exposed by the provider. Raises on connection / HTTP errors."""
    if settings.provider == "ollama":
        from ollama import Client

        client = Client(host=settings.base_url, headers=settings.headers, timeout=10)
        return {m.model for m in client.list().models if m.model}

    from openai import OpenAI

    # Local OpenAI compatible servers (LM Studio, llama.cpp, vLLM) may not need a key
    client = OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key or "not-needed",
        timeout=10,
        max_retries=0,
    )
    return {m.id for m in client.models.list()}


def check_connection(settings: ProviderSettings, model_name: str) -> None:
    """Fail fast if the server is unreachable; warn if the model is not on it."""
    if settings.provider == "openai_compatible" and not settings.api_key:
        # Model listing is often public, so a missing key would only show up later
        print(
            f"[WARNING] {settings.env_prefix}_API_KEY is empty: fine for local "
            "servers, but OpenRouter / Requesty will reject every request."
        )

    try:
        available = list_models(settings)
    except Exception as e:
        raise ConnectionError(
            f"Cannot reach {settings.provider} at {settings.base_url}: {e}. "
            f"Check {settings.env_prefix}_BASE_URL / {settings.env_prefix}_API_KEY "
            "in your .env file."
        ) from e

    if not model_is_available(model_name, available):
        hint = (
            ", ".join(sorted(available))
            if len(available) <= 20
            else f"{len(available)} models, check the provider's model list"
        )
        print(
            f"[WARNING] Model '{model_name}' not found on {settings.base_url}. "
            f"Available: {hint or 'none'}"
        )


class LLMManager:
    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float | None,
        template: str,
        thinking_enabled: bool,
        settings: ProviderSettings | None = None,
    ) -> None:
        self.settings: ProviderSettings = settings or ProviderSettings.from_env(
            provider
        )
        if self.settings.provider != provider:
            raise ValueError(
                f"Provider mismatch: {provider} != {self.settings.provider}"
            )

        self.provider: str = provider
        self.model_name: str = model_name
        self.temperature: float | None = temperature  # None = provider default
        self.thinking_enabled: bool = thinking_enabled
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
                base_url=self.settings.base_url,
                client_kwargs={"headers": self.settings.headers},
            )
        elif self.provider == "openai_compatible":
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                base_url=self.settings.base_url,
                api_key=self.settings.api_key or "not-needed",
                max_retries=2,  # also covers transient 429 rate limits
            )
        else:
            raise ValueError(f"Unable to initialize the model for {self.provider}.")

        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(template)
        # StrOutputParser turns chat messages (OpenAI) into plain text, like OllamaLLM
        return prompt | model | StrOutputParser()

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
