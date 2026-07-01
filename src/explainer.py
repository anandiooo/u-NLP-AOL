import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

from logicheck.model import FALLACY_DEFINITIONS

SYSTEM_PROMPT = (
    "You are a critical thinking educator specializing in logical fallacies. "
    "Your task is to explain logical fallacies in clear, educational language for a general audience. "
    "Be concise (2-4 sentences), specific to the given text, and non-judgmental in tone. "
    "Always explain WHAT the fallacy is and HOW it appears in the specific text."
)

# abstract base class for explainers
class BaseExplainer(ABC):
    @property
    @abstractmethod
    # returns the name
    def name(self):
        ...

    @abstractmethod
    # explains the detected fallacy
    def explain(self, text, fallacy_labels):
        ...

# explains fallacy using predefined templates
class TemplateExplainer(BaseExplainer):
    @property
    # returns the name
    def name(self):
        return "Local Template"

    # explains the detected fallacy
    def explain(self, text, fallacy_labels):
        labels = [l for l in fallacy_labels if l != "no_fallacy"]
        if not labels:
            return "No logical fallacy detected. The reasoning appears valid."
        parts = []
        for label in labels:
            name = label.replace("_", " ").title()
            definition = FALLACY_DEFINITIONS.get(label, "a logical error")
            parts.append(f"This text employs a **{name}**, which involves {definition}.")
        return " ".join(parts)

# explains fallacy using gemini api
class GeminiExplainer(BaseExplainer):
    # initialises the class instance
    def __init__(self, model_name="gemini-3.1-pro", max_tokens=256, temperature=0.3):
        self.model_name = model_name
        self._name = f"Gemini {model_name.split('-')[1]}" if "-" in model_name else "Gemini"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self._fallback = TemplateExplainer()
        self._init_client()

    @property
    # returns the name
    def name(self):
        return self._name if self._client else self._fallback.name

    # initialises the api client
    def _init_client(self):
        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment variables.")
                raise EnvironmentError("GEMINI_API_KEY not set in environment.")
            else:
                logger.info(f"API key found: {api_key[:4]}...")
            self._client = genai.Client(api_key=api_key)
            logger.info(f"Gemini explainer initialised: {self.model_name}")
        except Exception as e:
            logger.exception(f"Could not initialise Gemini client: {e}. Using template fallback.")
            self._client = None

    # builds the prompt for the llm
    def _build_prompt(self, text, labels):
        defs = "\n".join(
            f"- {l.replace('_', ' ').title()}: {FALLACY_DEFINITIONS.get(l, 'a logical error')}"
            for l in labels
        )
        return (
            f"The following text has been detected as containing these logical fallacies:\n{defs}\n\n"
            f"Text: \"{text}\"\n\n"
            f"Explain clearly and educationally how this text demonstrates these fallacies. "
            f"Be specific and reference the text directly."
        )

    # explains the detected fallacy
    def explain(self, text, fallacy_labels):
        labels = [l for l in fallacy_labels if l != "no_fallacy"]
        if not labels:
            return "No logical fallacy detected. The reasoning appears valid."
        if self._client is None:
            logger.warning("Gemini client is not initialised. Falling back to template.")
            return self._fallback.explain(text, fallacy_labels)

        prompt = self._build_prompt(text, labels)
        try:
            from google.genai import types
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            )

            if response and response.text:
                return response.text.strip()
            else:
                logger.error("Gemini returned an empty response.")
                return self._fallback.explain(text, fallacy_labels)

        except Exception as e:
            logger.error(f"Gemini API error: {e}. Falling back to template.")
            return self._fallback.explain(text, fallacy_labels)

# factory function to create explainer
def build_explainer(config):
    explainer_cfg = config.get("explainer", {})
    provider = explainer_cfg.get("provider", "template")
    if provider == "gemini":
        return GeminiExplainer(
            model_name=explainer_cfg.get("model", "gemini-3.1-pro"),
            max_tokens=explainer_cfg.get("max_output_tokens", 256),
            temperature=explainer_cfg.get("temperature", 0.3),
        )
    else:
        logger.info("Using TemplateExplainer (no API).")
        return TemplateExplainer()
