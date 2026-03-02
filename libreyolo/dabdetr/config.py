"""
Training configuration for DAB-DETR.

Provides a dataclass-based configuration with DAB-DETR-specific defaults.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Union
import yaml


@dataclass
class DABDETRTrainConfig:
    """Configuration for DAB-DETR training."""

    # Model configuration
    size: str = "r50"  # "r50", "r50-dc5", "r50-3pat", "r50-dc5-3pat"
    num_classes: int = 80
    pretrained: Optional[str] = None

    # Architecture hyperparameters (user-overridable)
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    num_queries: int = 300
    modulate_hw_attn: bool = True

    # Data configuration
    data: Optional[str] = None  # Path to data.yaml
    data_dir: Optional[str] = None  # Direct path to dataset
    imgsz: int = 640

    # Training parameters
    epochs: int = 50
    batch: int = 4
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Optimizer settings
    optimizer: str = "adamw"
    lr0: float = 0.0001
    lr_backbone: float = 0.00001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    betas: Tuple[float, float] = (0.9, 0.999)
    nesterov: bool = False

    # Learning rate schedule
    scheduler: str = "linear"  # "linear", "cos", "warmcos"
    warmup_epochs: int = 0
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.01

    # Augmentation settings
    mosaic_prob: float = 0.5
    mixup_prob: float = 0.0
    hsv_prob: float = 0.1
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0

    # Training features
    ema: bool = True
    ema_decay: float = 0.9999
    amp: bool = True

    # Checkpointing
    project: str = "runs/train"
    name: str = "dabdetr_train"
    exist_ok: bool = False
    save_period: int = 10
    eval_interval: int = 1

    # Workers
    workers: int = 4

    # Early stopping
    patience: int = 50

    # Resume
    resume: bool = False

    # Logging
    log_interval: int = 1

    # Reproducibility
    seed: int = 0

    def __post_init__(self):
        valid_sizes = ["r50", "r50-dc5", "r50-3pat", "r50-dc5-3pat"]
        if self.size not in valid_sizes:
            raise ValueError(
                f"Invalid size '{self.size}'. Must be one of {valid_sizes}"
            )

        valid_schedulers = ["linear", "cos", "warmcos"]
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"Invalid scheduler '{self.scheduler}'. Must be one of {valid_schedulers}"
            )

        valid_optimizers = ["sgd", "adam", "adamw"]
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer '{self.optimizer}'. Must be one of {valid_optimizers}"
            )

        if isinstance(self.mosaic_scale, list):
            self.mosaic_scale = tuple(self.mosaic_scale)
        if isinstance(self.mixup_scale, list):
            self.mixup_scale = tuple(self.mixup_scale)
        if isinstance(self.betas, list):
            self.betas = tuple(self.betas)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DABDETRTrainConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, **kwargs) -> "DABDETRTrainConfig":
        data = asdict(self)
        data.update(kwargs)
        return DABDETRTrainConfig(**data)

    @property
    def input_size(self) -> Tuple[int, int]:
        return (self.imgsz, self.imgsz)

    @property
    def effective_lr(self) -> float:
        return self.lr0 * self.batch / 16.0

    def __repr__(self) -> str:
        lines = ["DABDETRTrainConfig("]
        for key, value in asdict(self).items():
            lines.append(f"  {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)


# Preset configurations — variant-specific overrides for backbone_dilation and num_patterns
DAB_DETR_PRESETS = {
    "r50": {"backbone_dilation": False, "num_patterns": 0},
    "r50-dc5": {"backbone_dilation": True, "num_patterns": 0},
    "r50-3pat": {"backbone_dilation": False, "num_patterns": 3},
    "r50-dc5-3pat": {"backbone_dilation": True, "num_patterns": 3},
}

# Default train configs per variant
DABDETR_CONFIGS = {size: DABDETRTrainConfig(size=size) for size in DAB_DETR_PRESETS}


def get_dabdetr_config(size: str = "r50", **kwargs) -> DABDETRTrainConfig:
    """Get a preset DAB-DETR training configuration with optional overrides.

    Args:
        size: Variant code ("r50", "r50-dc5", "r50-3pat", "r50-dc5-3pat").
        **kwargs: Override any configuration parameter.

    Returns:
        DABDETRTrainConfig instance.
    """
    if size not in DABDETR_CONFIGS:
        raise ValueError(
            f"Unknown size '{size}'. Available: {list(DABDETR_CONFIGS.keys())}"
        )
    config = DABDETR_CONFIGS[size]
    if kwargs:
        config = config.update(**kwargs)
    return config
