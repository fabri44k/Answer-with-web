from langchain.prompts import ChatPromptTemplate
import re

class LLMManager:

    __SUPPORTED_PROVIDERS = ["ollama"]

    def __init__(self, provider, model_name, temperature, template, thinking_enabled):
        
        if provider not in self.__SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers are: {self.__SUPPORTED_PROVIDERS}")
        
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.thinking_enabled = thinking_enabled
        self.chain = self.__initialize_llm(template)
        

    def __initialize_llm(self, template):
        
        if not template:
            raise ValueError("Template cannot be empty. Please provide a valid template.")

        if self.provider == "ollama":
            from langchain_ollama import OllamaLLM
            model = OllamaLLM(model=self.model_name, temperature=self.temperature)

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        return chain

    # remove <think>...</think> content from the text
    def __remove_thinking_from_text(self, text):
        if not text:
            return text
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def answer_query(self, dict_for_template):
        
        if not dict_for_template or not isinstance(dict_for_template, dict):
            raise ValueError("Input must be a dictionary with the required keys for the template.")
        
        try:
            response = self.chain.invoke(dict_for_template)
        except Exception as e:
            raise RuntimeError(f"Error invoking the LLM chain: {e}. Check the provider and the template are correctly formatted.")

        if self.thinking_enabled:
            return self.__remove_thinking_from_text(response)
        
        return response.strip()