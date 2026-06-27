from __future__ import annotations

import logging
from typing import Any

from sdgsystem.configs.config import TextDistillConfig
from sdgsystem.distillation.evol_instruct import EvolInstructDistillation
from sdgsystem.distillation.sdg_distill import SDGDistillation
from sdgsystem.distillation.self_instruct import SelfInstructDistillation

from .config import SynAgenticDataConfig, to_dataarc_demo

logger = logging.getLogger(__name__)


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
    try:
        if strategy == "evol_instruct":
            return _generate_evol_directional_samples(generator, seed_item=seed_item, config=config)
        return generator.generate(demo_examples=[to_dataarc_demo(seed_item)])
    except Exception as exc:
        logger.warning("DataArc %s candidate generation failed for %s: %s", strategy, domain, exc)
        return []


def _generate_evol_directional_samples(
    generator: EvolInstructDistillation,
    *,
    seed_item: dict[str, Any],
    config: SynAgenticDataConfig,
) -> list[dict[str, Any]]:
    demo_examples = [to_dataarc_demo(seed_item)]
    samples: list[dict[str, Any]] = []
    for direction in config.evol_directions:
        direction_samples: list[dict[str, Any]] = []
        for attempt in range(1, config.candidate_generation_attempts + 1):
            try:
                prompt = generator._build_batch_prompt(
                    demo_examples=demo_examples,
                    batch_size=config.num_samples_per_strategy,
                    evol_direction=direction,
                )
                response = generator.model.generate(
                    prompt,
                    temperature=generator.config.temperature,
                    max_tokens=4096,
                    n=1,
                )
                direction_samples = [
                    sample
                    for sample in generator._parse_batch_response(response)
                    if isinstance(sample, dict) and "input" in sample and "output" in sample
                ]
            except Exception as exc:
                logger.warning("Evol-Instruct %s candidate generation failed: %s", direction, exc)
                direction_samples = []
            if direction_samples:
                break
            logger.warning("Evol-Instruct %s candidate generation attempt %s produced no parseable samples", direction, attempt)
        if not direction_samples:
            direction_samples = [_evol_seed_fallback(seed_item, direction)]
        for sample in direction_samples[: config.num_samples_per_strategy]:
            annotated = dict(sample)
            metadata = dict(annotated.get("metadata") or {})
            metadata["evol_direction"] = direction
            metadata.setdefault("candidate_source", "evol_instruct")
            annotated["metadata"] = metadata
            samples.append(annotated)
    return samples


def _evol_seed_fallback(seed_item: dict[str, Any], direction: str) -> dict[str, Any]:
    return {
        "input": str(seed_item["question"]),
        "output": f"Rationale:\n{seed_item['rationale']}\n\nFinal Answer: {seed_item['final_answer']}",
        "metadata": {"evol_direction": direction, "candidate_source": "seed_fallback"},
    }


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
