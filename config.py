from dataclasses import dataclass, field
from typing import Optional
import datetime


@dataclass
class OptimizerConfig:
    lr: float = field(default=1e-4)

@dataclass
class Config:
    exp_name: str = field(default_factory=datetime.datetime.now)
    block_size: int = field(default=8) # почему-то не работает увеличение параметра
    batch_size: int = field(default=2048)
    train_test_split_ratio: float = field(default=0.8)
    vocab_size: Optional[int] = field(default=None)
    max_epochs: int = field(default=100)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    token_embedding_size: int = field(default=256)
    pos_embedding_size: int = field(default=256)
    head_size: int = field(default=16)
    num_heads: int = field(default=4)
    repeat_blocks: int = field(default=3)
    max_tokens_generation: int = field(default=100)
    num_generated_phrases: int = 10