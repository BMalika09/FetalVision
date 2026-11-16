import wandb

def start_run(config: dict):
    """Initialize a W&B run with the experiment configuration."""

    tracking_config = config["tracking"]

    if not tracking_config.get("enabled", False):
        return None

    return wandb.init(
        project=tracking_config["project"],
        name=tracking_config.get("run_name"),
        config=config,
    )


def log_epoch(run, epoch: int, metrics: dict):
    """Log training and validation metrics for one epoch."""

    if run is not None:
        run.log({"epoch": epoch, **metrics})


def finish_run(run):
    """Finish the current W&B run."""

    if run is not None:
        run.finish()
