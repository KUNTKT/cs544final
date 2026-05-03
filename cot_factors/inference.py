from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class HFEngine:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> "HFEngine":
        dtype = DTYPE_MAP.get(torch_dtype, torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        return cls(model=model, tokenizer=tok)

    def build_chat_prompt(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback: concatenate when no chat template
        return user_text

    @torch.inference_mode()
    def generate_one(
        self,
        user_text: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> str:
        prompt = self.build_chat_prompt(user_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p

        out = self.model.generate(**inputs, **gen_kwargs)
        # Decode only the newly generated tokens
        in_len = inputs["input_ids"].shape[1]
        new_ids = out[0, in_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return text


def optional_vllm_engine(model_name: str):
    """Optional vLLM hook; not required by default."""
    return None
