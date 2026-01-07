import os

def get_llm(logger=None):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, temperature=0.2)
            if logger: logger.info(f"LLM: OpenAI ChatOpenAI('{model}')")
            return llm
        except Exception as e:
            if logger: logger.warning(f"OpenAI LLM init failed: {e}")

    try:
        from transformers import pipeline
        gen = pipeline(
            "text-generation",
            model=os.getenv("LOCAL_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            device_map="auto",
            max_new_tokens=256,
        )
        class _Local:
            def invoke(self, prompt: str):
                return gen(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
        if logger: logger.info("LLM: local transformers pipeline")
        return _Local()
    except Exception as e:
        if logger: logger.warning(f"Local LLM unavailable: {e}. Using extractive fallback.")

    class _Extractive:
        def invoke(self, prompt: str):
            return "Не удалось инициализировать LLM. Ниже — извлечённый контекст:\n\n" + prompt[-1500:]
    return _Extractive()
