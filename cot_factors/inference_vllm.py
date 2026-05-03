"""vLLM batched inference (matches HFEngine: build_chat_prompt + generate_one / generate_batch)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, List, Optional

from tqdm import tqdm

from .config import VllmConfig


@dataclass
class VLLMEngine:
    llm: Any
    tokenizer: Any

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        vllm_cfg: VllmConfig,
        trust_remote_code: bool = True,
    ) -> "VLLMEngine":
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. In your env (e.g. sspo): pip install vllm"
            ) from e

        llm = LLM(
            model=model_name,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=vllm_cfg.tensor_parallel_size,
            gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
            dtype=vllm_cfg.dtype,
            max_model_len=vllm_cfg.max_model_len,
            enforce_eager=vllm_cfg.enforce_eager,
        )
        tok = llm.get_tokenizer()
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            tok.pad_token = tok.eos_token
        return cls(llm=llm, tokenizer=tok)

    def build_chat_prompt(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_text

    def generate_batch(
        self,
        user_texts: List[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        batch_size: int = 32,
        progress_desc: Optional[str] = None,
    ) -> List[str]:
        from vllm import SamplingParams

        # Greedy: temperature=0
        temp = max(temperature, 1e-5) if do_sample else 0.0
        sp = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p if do_sample else 1.0,
        )
        prompts: List[str] = [self.build_chat_prompt(u) for u in user_texts]
        out_texts: List[str] = []
        bs = max(1, batch_size)
        n = len(prompts)
        desc = progress_desc or "vLLM generate"
        pbar = tqdm(
            total=n,
            desc=desc[:60] + ("…" if len(desc) > 60 else ""),
            unit="item",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=False,
        )
        try:
            for i in range(0, n, bs):
                chunk = prompts[i : i + bs]
                outputs = self.llm.generate(chunk, sp)
                out_texts.extend(o.outputs[0].text for o in outputs)
                pbar.update(len(chunk))
        finally:
            pbar.close()
        return out_texts

    def generate_one(
        self,
        user_text: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> str:
        return self.generate_batch(
            [user_text],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            batch_size=1,
            progress_desc=None,
        )[0]
