import re
from typing import Any

from langchain.prompts import ChatPromptTemplate


class LLMManager:
    __SUPPORTED_PROVIDERS: list[str] = ["ollama"]

    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float,
        template: str,
        thinking_enabled: bool,
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
        self.chain = self.__initialize_llm(template)

    def __initialize_llm(self, template: str) -> Any:
        if not template:
            raise ValueError(
                "Template cannot be empty. Please provide a valid template."
            )

        if self.provider == "ollama":
            from langchain_ollama import OllamaLLM

            model = OllamaLLM(model=self.model_name, temperature=self.temperature)
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
