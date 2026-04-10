import itertools
import logging
import os
import re
import shutil
import warnings
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from ..lightning import BaseLightningModule, DataModule
from ..utils import load_hydra_config, seed_all

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = typer.Typer()


@app.command()
def train(
    config_file: Annotated[str, typer.Argument(help="Path to YAML config file.")],
    overrides: Annotated[list[str] | None, typer.Argument(help="Overrides for the config file.")] = None,
    overwrite: Annotated[bool, typer.Option(help="Whether to overwrite existing log directory.")] = False,
    resume: Annotated[bool, typer.Option(help="Whether to resume training from the last checkpoint.")] = False,
    debug: Annotated[bool, typer.Option(help="Whether to set logging level to DEBUG.")] = False,
    select: Annotated[str | None, typer.Option(help="Name of sweep configuration to train.")] = None,
    wandb: Annotated[bool, typer.Option(help="Whether to log to Weights & Biases.")] = False,
    log_model: Annotated[bool, typer.Option(help="Whether to upload checkpoints to Weights & Biases.")] = False,
) -> None:
    """Train a model."""
    if not os.path.exists(config_file):
        raise FileNotFoundError("Config file does not exist.")
    if not os.path.isfile(config_file):
        raise ValueError("Input config must be a file.")
    if not config_file.endswith(".yaml"):
        raise ValueError("Config file must be a YAML file.")
    if overwrite and resume:
        raise ValueError("Cannot use both --overwrite and --resume.")

    warnings.filterwarnings("ignore", "The .* does not have many workers")
    warnings.filterwarnings("ignore", "The number of training batches .* is smaller than the logging interval")
    warnings.filterwarnings("ignore", "Experiment logs directory .* exists and is not empty")
    warnings.filterwarnings("ignore", ".* is set, but there is no last checkpoint available")
    warnings.filterwarnings("ignore", "Checkpoint directory .* exists and is not empty")

    load_dotenv()

    base_cfg, config_name = load_hydra_config(config_file, overrides=overrides)

    config_name = base_cfg.get("name", config_name)

    logging.basicConfig(format="%(levelname)s: %(message)s")
    if debug:
        logging.getLogger("addse").setLevel(logging.DEBUG)

    cfgs: dict[str, DictConfig] = {}
    if (sweep_list := base_cfg.get("sweep")) is None:
        cfgs[config_name] = base_cfg
    else:
        if not isinstance(sweep_list, ListConfig) or not all(isinstance(s, DictConfig) for s in sweep_list):
            raise ValueError("Sweep configuration must be a list of mappings from names to updates.")
        product_sweeps = {}
        for sweep_items in itertools.product(*[s.items() for s in sweep_list]):
            product_name = "-".join(f"{s[0]}" for s in sweep_items)
            product_update = {k: v for d in sweep_items for k, v in d[1].items()}
            product_sweeps[product_name] = DictConfig(product_update)
        if len(product_sweeps.values()) != len(set(product_sweeps.values())):
            raise ValueError("Found duplicate sweep parameters sets after product.")
        for sweep_name, sweep_update in product_sweeps.items():
            merged_cfg = OmegaConf.merge(base_cfg, sweep_update)
            assert isinstance(merged_cfg, DictConfig)
            cfgs[f"{config_name}-{sweep_name}"] = merged_cfg
        logger.info(f"Found {len(cfgs)} sweep configurations: {', '.join(cfgs.keys())}")

    if select is not None and select not in cfgs:
        raise ValueError(f"Selected configuration '{select}' not found in sweep configurations.")

    for name, cfg in cfgs.items():
        if select is not None and name != select:
            continue

        logger.info(f"Training: {name}")

        default_loggers: list[dict[str, Any]] = [
            {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": "logs",
                "name": name,
                "version": "",
            }
        ]

        if wandb:
            if "logger" in cfg.trainer:
                raise ValueError("Explicit logger YAML configuration is incompatible with --wandb.")
            run_id = None
            if resume:
                # infer wandb run from existing log directory
                run_ids = [re.match(r"run-\d+_\d+-(\w+)", d) for d in os.listdir(os.path.join("logs", name, "wandb"))]
                run_id_set = {m.group(1) for m in run_ids if m is not None}
                if len(run_id_set) == 0:
                    raise ValueError("No existing W&B run found to resume from.")
                if len(run_id_set) > 1:
                    raise ValueError("Multiple existing W&B runs found to resume from.")
                run_id = run_id_set.pop()
                logger.info(f"Resuming W&B run: {run_id}")
            default_loggers.append(
                {
                    "_target_": "lightning.pytorch.loggers.WandbLogger",
                    "save_dir": os.path.join("logs", name),
                    "name": name,
                    "id": run_id,
                    "log_model": log_model,
                }
            )

        with open_dict(cfg):
            cfg.trainer.setdefault("logger", default_loggers)
            checkpoint_monitor = cfg.get("checkpoint_monitor", "val_loss")
            checkpoint_mode = cfg.get("checkpoint_mode", "min")
            checkpoint_filename = cfg.get("checkpoint_filename", f"{{epoch:02d}}-{{{checkpoint_monitor}:.2f}}")
            cfg.trainer.setdefault(
                "callbacks",
                [
                    {
                        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "monitor": checkpoint_monitor,
                        "save_top_k": 1,
                        "mode": checkpoint_mode,
                        "filename": checkpoint_filename,
                    },
                    {
                        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "monitor": "epoch",
                        "save_top_k": 1,
                        "mode": "max",
                        "filename": "last",
                    },
                    {
                        "_target_": "lightning.pytorch.callbacks.TQDMProgressBar",
                        "refresh_rate": 1,
                        "leave": True,
                    },
                    {
                        "_target_": "addse.callbacks.TimerCallback",
                    },
                    {
                        "_target_": "addse.callbacks.GPUMemoryCallback",
                    },
                ],
            )
            cfg.trainer.callbacks += cfg.get("extra_callbacks", [])

        seed_all(cfg.seed)

        lm: BaseLightningModule = instantiate(cfg.lm)
        dm: DataModule = instantiate(cfg.dm)
        trainer: Trainer = instantiate(cfg.trainer)

        if trainer.logger is not None:
            log_dir = trainer.logger.log_dir
            if log_dir is not None and os.path.isdir(log_dir) and os.listdir(log_dir):
                if overwrite:
                    logger.info(f"Overwriting existing log directory: {log_dir}")
                    shutil.rmtree(log_dir)
                elif not resume:
                    raise ValueError(
                        f"Log directory '{log_dir}' already exists and is not empty. Use --overwrite or --resume."
                    )

        for logger_ in trainer.loggers:
            logger_.log_hyperparams(
                {
                    "seed": cfg.seed,
                    "lm": OmegaConf.to_container(cfg.lm, resolve=True),
                    "dm": OmegaConf.to_container(cfg.dm, resolve=True),
                    "trainer": OmegaConf.to_container(cfg.trainer, resolve=True),
                }
            )

        trainer.fit(lm, dm, ckpt_path="last" if resume else None)

        ckpt_callback = trainer.checkpoint_callback
        ckpt_path = None if ckpt_callback is None else getattr(ckpt_callback, "best_model_path", None)
        ckpt_path = None if ckpt_path == "" else ckpt_path
        has_test_data = getattr(dm, "test_dset", None) is not None and getattr(dm, "test_dataloader_fn", None) is not None
        if has_test_data:
            trainer.test(lm, dm, ckpt_path)
        else:
            logger.info("Skipping trainer.test because no test dataset/dataloader is configured.")

        auto_eval_cfg = cfg.get("auto_eval")
        auto_eval_enabled = bool(auto_eval_cfg) if isinstance(auto_eval_cfg, bool) else bool(auto_eval_cfg.get("enabled", False)) if auto_eval_cfg else False
        if auto_eval_enabled:
            from .eval import eval as eval_command

            eval_overrides = overrides
            eval_select = name if base_cfg.get("sweep") is not None else None
            eval_output_db = os.path.join("logs", name, "auto_eval.db")
            if isinstance(auto_eval_cfg, DictConfig):
                eval_output_db = str(auto_eval_cfg.get("output_db", eval_output_db))
                eval_overwrite = bool(auto_eval_cfg.get("overwrite", True))
                eval_num_examples = auto_eval_cfg.get("num_examples")
                eval_compute_loss = bool(auto_eval_cfg.get("compute_loss", False))
                eval_return_nfe = bool(auto_eval_cfg.get("return_nfe", False))
                eval_clean_input = bool(auto_eval_cfg.get("clean_input", False))
                eval_no_metrics = bool(auto_eval_cfg.get("no_metrics", False))
            else:
                eval_overwrite = True
                eval_num_examples = None
                eval_compute_loss = False
                eval_return_nfe = False
                eval_clean_input = False
                eval_no_metrics = False

            if ckpt_path is None:
                logger.warning("Skipping auto-eval because no best checkpoint was found.")
            else:
                eval_command(
                    config_file,
                    ckpt_path,
                    overrides=eval_overrides,
                    select=eval_select,
                    output_db=eval_output_db,
                    overwrite=eval_overwrite,
                    num_examples=eval_num_examples,
                    clean_input=eval_clean_input,
                    return_nfe=eval_return_nfe,
                    compute_loss=eval_compute_loss,
                    no_metrics=eval_no_metrics,
                )
