"""Hugging Face open-source model client."""

from __future__ import annotations

import os

from models.base import ChatMessage


class HuggingFaceOSSClient:
    """Lazy-loaded local Transformers client for small instruct models."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("OSS_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
        self.name = self.model_id
        self._pipeline = None
        self._tokenizer = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        device = self._best_device(torch)
        dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
            device=device,
        )
        return self._pipeline

    @staticmethod
    def _best_device(torch):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _format_messages(self, messages: list[ChatMessage]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded.")

        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        rendered = []
        for message in messages:
            role = message["role"].upper()
            rendered.append(f"{role}: {message['content']}")
        rendered.append("ASSISTANT:")
        return "\n".join(rendered)

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        pipe = self._load_pipeline()
        prompt = self._format_messages(messages)
        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            return_full_text=False,
            pad_token_id=self._tokenizer.eos_token_id if self._tokenizer else None,
        )
        return outputs[0]["generated_text"].strip()
