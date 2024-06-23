from dataclasses import dataclass, field
from typing import Optional
import datetime


@dataclass
class OptimizerConfig:
    lr: float = field(default=1e-3)

@dataclass
class Config:
    exp_name: str = field(default_factory=datetime.datetime.now)
    block_size: int = field(default=3)
    batch_size: int = field(default=2048)
    train_test_split_ratio: float = field(default=0.8)
    vocab_size: Optional[int] = field(default=None)
    max_epochs: int = field(default=100)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)