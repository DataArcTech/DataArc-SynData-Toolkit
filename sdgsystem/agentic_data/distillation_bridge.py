from __future__ import annotations

from typing import Any

from sdgsystem.configs.config import TextDistillConfig
from sdgsystem.distillation.evol_instruct import EvolInstructDistillation
from sdgsystem.distillation.sdg_distill import SDGDistillation
from sdgsystem.distillation.self_instruct import SelfInstructDistillation

from .config import SynAgenticDataConfig, to_dataarc_demo


def get_distillation_class(strategy: str):
    if strategy == "few_shot":
        return SDGDistillation
    if strategy == "self_instruct":
        return SelfInstructDistillation
    if strategy == "evol_instruct":
        return EvolInstructDistillation
    raise ValueError(f"Unsupported synthesis strategy: {strategy}")


def build_text_distill_config(domain: str, strategy: str, base_config: SynAgenticDataConfig) -> TextDistillConfig:
    return TextDistillConfig(
        name=f"syn_agentic_{strategy}_{domain}",
        domain=domain,
        demo_examples_path=None,
        task_instruction=_task_instruction(domain, strategy),
        input_instruction=_input_instruction(domain),
        output_instruction=_output_instruction(),
        num_samples=base_config.num_samples_per_strategy,
        batch_size=base_config.num_samples_per_strategy,
        temperature=base_config.temperature,
    )


def generate_candidate_samples(
    *,
    strategy: str,
    domain: str,
    seed_item: dict[str, Any],
    config: SynAgenticDataConfig,
    model_client: Any,
) -> list[dict[str, Any]]:
    distill_config = build_text_distill_config(domain, strategy, config)
    generator_cls = get_distillation_class(strategy)
    generator = generator_cls(model=model_client, config=distill_config)
    return generator.generate(demo_examples=[to_dataarc_demo(seed_item)])


def _task_instruction(domain: str, strategy: str) -> str:
    return (
        f"Generate {domain} training examples for a verifiable agentic reasoning dataset. "
        f"The generation strategy is {strategy}. The example should be self-contained, "
        "computationally checkable, and suitable for later conversion into executable Python rationale form."
    )


def _input_instruction(domain: str) -> str:
    return f"The input should be a self-contained {domain} question or task."


def _output_instruction() -> str:
    return "The output should include enough reasoning or answer context for a later verifier-backed completion step."
