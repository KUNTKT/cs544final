from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    path: str = ""
    max_items: Optional[int] = 200
    seed: int = 42


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class SelfConsistencyConfig:
    enabled: bool = False
    num_samples: int = 5
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class OutputConfig:
    dir: str = "./outputs"


@dataclass
class VllmConfig:
    """Used only when backend=vllm."""

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    max_model_len: int = 4096
    batch_size: int = 32
    enforce_eager: bool = False


@dataclass
class ExperimentConfig:
    backend: str = "hf"  # hf | vllm
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    self_consistency: SelfConsistencyConfig = field(default_factory=SelfConsistencyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)


def _merge_dataclass(cls, d: dict[str, Any]) -> Any:
    field_names = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in d.items() if k in field_names}
    return cls(**kwargs)


def load_yaml(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    backend = str(raw.get("backend", "hf"))
    return ExperimentConfig(
        backend=backend,
        data=_merge_dataclass(DataConfig, raw.get("data", {})),
        model=_merge_dataclass(ModelConfig, raw.get("model", {})),
        generation=_merge_dataclass(GenerationConfig, raw.get("generation", {})),
        self_consistency=_merge_dataclass(SelfConsistencyConfig, raw.get("self_consistency", {})),
        output=_merge_dataclass(OutputConfig, raw.get("output", {})),
        vllm=_merge_dataclass(VllmConfig, raw.get("vllm", {})),
    )


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    p.add_argument("--backend", type=str, choices=["hf", "vllm"], default=None, help="Inference backend: hf or vllm")
    p.add_argument("--data-path", type=str, default=None, help="Override data.path")
    p.add_argument("--model", type=str, default=None, help="Override model.name")
    p.add_argument("--max-items", type=int, default=None, help="Override data.max_items")
    p.add_argument("--output-dir", type=str, default=None, help="Override output.dir")
    p.add_argument("--seed", type=int, default=None, help="Override data.seed")


def apply_cli_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    if getattr(args, "backend", None):
        cfg.backend = args.backend
    if args.data_path:
        cfg.data.path = args.data_path
    if args.model:
        cfg.model.name = args.model
    if args.max_items is not None:
        cfg.data.max_items = args.max_items
    if args.output_dir:
        cfg.output.dir = args.output_dir
    if args.seed is not None:
        cfg.data.seed = args.seed
