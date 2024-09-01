from dataclasses import dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class KBData:
    """Knowledgebase data (atomic + conceptnet)"""

    atomic_file: str
    conceptnet_file: str
    tokenization_config: Dict[str, Any]


@dataclass
class ROCData:
    """ROC Stories dataset"""

    file: str
    tokenization_config: Dict[str, Any]


@dataclass
class ModelConfig:
    pretrained_model: str
    added_special_tokens: Dict[str, str] = field(default_factory=dict)
    added_tokens: List[str] = field(default_factory=list)
