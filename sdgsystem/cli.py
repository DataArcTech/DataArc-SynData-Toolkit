import click
from dotenv import load_dotenv

from .pipeline import Pipeline
from .configs.config import SDGSConfig

load_dotenv()


def transform_arguments_to_configurations(
    output_dir: str | None,
    export_format: str | None,
    task_definition: str | None,
    generator_provider: str | None,
    generator_model: str | None,
) -> SDGSConfig:
    config = {}

    # global settings
    if output_dir:
        config["output_dir"] = output_dir
    if export_format:
        config["export_format"] = export_format

    # task - inject task_definition into existing modality structure
    # The modality (text.local, text.web, text.distill) is determined by the config file
    if task_definition:
        # This will be merged with the config file settings
        # The actual modality is determined by which config section is present
        config["task"] = {
            "text": {
                "local": {"generation": {"task_instruction": task_definition}},
                "web": {"task_instruction": task_definition},
                "distill": {"task_instruction": task_definition}
            }
        }

    # generator
    llm_config = {}
    if generator_provider:
        llm_config["provider"] = generator_provider
    if generator_model:
        llm_config["model"] = generator_model

    if llm_config:
        config["llm"] = llm_config

    return config


@click.group()
def cli():
    """DataArc Synthetic Data Generation Toolkit."""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True), required=False)
@click.option("--output_dir", type=str, help="Output directory for generated data")
@click.option("--export_format", type=str, help="Export format (jsonl, json, csv)")
@click.option("--task_definition", type=str, help="Task definition/instruction")
@click.option("--generator_provider", type=str, help="LLM provider for generator (e.g., openai, ollama)")
@click.option("--generator_model", type=str, help="LLM model name for generator")
def generate(
    config_file: str | None,
    output_dir: str | None,
    export_format: str | None,
    task_definition: str | None,
    generator_provider: str | None,
    generator_model: str | None,
) -> None:
    """Generate synthetic training data from a YAML configuration file."""
    # process configuration from a YAML file.
    config = SDGSConfig.from_yaml(config_file)

    # process additional options
    config.update(transform_arguments_to_configurations(
        output_dir,
        export_format,
        task_definition,
        generator_provider,
        generator_model,
    ))

    # run pipeline to get data
    pipeline = Pipeline(config)
    pipeline.run()


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--dataset", "-d", type=click.Path(exists=True), default=None,
              help="Override training dataset path")
@click.option("--model", type=str, default=None,
              help="Override model path")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Override output directory")
@click.option("--num-gpus", type=int, default=None,
              help="Override number of GPUs per node")
@click.option("--num-nodes", type=int, default=None,
              help="Override number of nodes")
@click.option("--verl-path", type=str, default=None,
              help="Path to verl installation (if not installed as package)")
def train(
    config_file: str,
    dataset: str | None,
    model: str | None,
    output: str | None,
    num_gpus: int | None,
    num_nodes: int | None,
    verl_path: str | None,
) -> None:
    """Train a model using a YAML configuration file.

    \b
    Example:
        sdg train configs/sft_example.yaml
        sdg train configs/grpo_example.yaml --num-gpus 4

    \b
    See configs/sft_example.yaml and configs/grpo_example.yaml for config examples.
    """
    from .trainer import load_training_config
    from .trainer.launcher import TrainingLauncher

    # Load config from YAML
    config = load_training_config(config_file)

    # Apply CLI overrides to nested config structure
    if dataset is not None:
        config.data.train_files = dataset
    if model is not None:
        config.model.path = model
    if output is not None:
        config.trainer.default_local_dir = output
    if num_gpus is not None:
        config.trainer.n_gpus_per_node = num_gpus
    if num_nodes is not None:
        config.trainer.nnodes = num_nodes

    # Get method as string
    method = config.method

    click.echo(f"Starting {method.upper()} training...")
    click.echo(f"  Train Dataset: {config.data.train_files}")
    click.echo(f"  Model: {config.model.path}")
    click.echo(f"  Output: {config.trainer.default_local_dir}")
    click.echo(f"  GPUs: {config.trainer.n_gpus_per_node} x {config.trainer.nnodes} nodes")

    try:
        launcher = TrainingLauncher(verl_path=verl_path)

        if method == "sft":
            return_code = launcher.run_sft(config)
        elif method == "grpo":
            return_code = launcher.run_grpo(config)
        else:
            raise NotImplementedError(f"Training method '{method}' not yet implemented")

        if return_code == 0:
            click.echo(click.style("Training completed successfully!", fg="green"))
        else:
            click.echo(click.style(f"Training failed with code {return_code}", fg="red"))
            raise SystemExit(return_code)

    except NotImplementedError as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        raise SystemExit(1)
    except Exception as e:
        click.echo(click.style(f"Training failed: {e}", fg="red"))
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
