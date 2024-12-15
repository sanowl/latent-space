from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any, Tuple, TypeVar, Protocol, Final
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import numpy as np
from enum import Enum, auto
from pathlib import Path
import json
from pydantic import BaseModel, Field

T = TypeVar('T')
TensorDict = Dict[str, torch.Tensor]

class DataFormat(Enum):
    CHAIN_OF_THOUGHT = auto()
    STEP_BY_STEP = auto()
    CONTINUOUS_THOUGHT = auto()

class ContinuousThoughtConfig(BaseModel):
    num_thoughts: int = Field(ge=1)
    thought_length: int = Field(ge=1)
    temperature: float = Field(ge=0.0, le=1.0)
    top_k: int = Field(ge=0)
    top_p: float = Field(ge=0.0, le=1.0)

@dataclass(frozen=True)
class ReasoningStep:
    text: str
    continuous_logits: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None

@dataclass(frozen=True)
class ReasoningExample:
    question: str
    answer: str  
    reasoning_steps: List[ReasoningStep]
    continuous_config: ContinuousThoughtConfig

class ContinuousThoughtModel(Protocol):
    def generate_continuous_thoughts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_thoughts: int,
        config: ContinuousThoughtConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

class CoconutDataset(Dataset):
    BOT_TOKEN: Final[str] = "<bot>"
    EOT_TOKEN: Final[str] = "<eot>"
    PAD_TOKEN: Final[str] = "<pad>"
    
    def __init__(
        self,
        examples: List[ReasoningExample],
        tokenizer: PreTrainedTokenizer,
        continuous_model: ContinuousThoughtModel,
        max_length: int,
        device: torch.device
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.continuous_model = continuous_model
        self.max_length = max_length
        self.device = device
        
        if not hasattr(tokenizer, "pad_token_id"):
            raise ValueError("Tokenizer must have pad_token_id")
        
        self._add_special_tokens()
        self._validate_examples()
        
    def _add_special_tokens(self) -> None:
        special_tokens: Dict[str, List[str]] = {
            "additional_special_tokens": [
                self.BOT_TOKEN,
                self.EOT_TOKEN,
                self.PAD_TOKEN
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
    def _validate_examples(self) -> None:
        for example in self.examples:
            if not example.reasoning_steps:
                raise ValueError("Examples must have reasoning steps")
            if not all(isinstance(step, ReasoningStep) for step in example.reasoning_steps):
                raise ValueError("All reasoning steps must be ReasoningStep objects")
                
    def _process_continuous_thoughts(
        self,
        example: ReasoningExample
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        processed_steps: List[Tuple[torch.Tensor, torch.Tensor]] = []
        
        for step in example.reasoning_steps:
            if step.continuous_logits is None:
                input_ids = self.tokenizer(
                    step.text,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )["input_ids"].to(self.device)
                
                attention_mask = torch.ones_like(input_ids)
                logits, mask = self.continuous_model.generate_continuous_thoughts(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_thoughts=example.continuous_config.num_thoughts,
                    config=example.continuous_config
                )
                processed_steps.append((logits, mask))
            else:
                processed_steps.append((step.continuous_logits, step.attention_mask))
                
        return processed_steps
        
    def _construct_input_sequence(
        self,
        example: ReasoningExample,
        continuous_thoughts: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> TensorDict:
        parts: List[str] = [example.question]
        
        for step, (logits, mask) in zip(example.reasoning_steps, continuous_thoughts):
            parts.append(f"{self.BOT_TOKEN}{step.text}{self.EOT_TOKEN}")
            
        parts.append(example.answer)
        
        encoded: TensorDict = self.tokenizer(
            " ".join(parts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        continuous_mask = self._create_continuous_mask(encoded["input_ids"])
        encoded["continuous_mask"] = continuous_mask
        encoded["continuous_logits"] = self._align_continuous_logits(
            continuous_thoughts, continuous_mask
        )
        
        return encoded
        
    def _create_continuous_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        bot_positions = input_ids == self.tokenizer.convert_tokens_to_ids(self.BOT_TOKEN)
        eot_positions = input_ids == self.tokenizer.convert_tokens_to_ids(self.EOT_TOKEN)
        
        mask = torch.cumsum(bot_positions, dim=-1) > torch.cumsum(eot_positions, dim=-1)
        return mask
        
    def _align_continuous_logits(
        self,
        continuous_thoughts: List[Tuple[torch.Tensor, torch.Tensor]],
        continuous_mask: torch.Tensor
    ) -> torch.Tensor:
        total_continuous_length = continuous_mask.sum().item()
        logits = torch.zeros(
            (len(continuous_thoughts), total_continuous_length, self.tokenizer.vocab_size),
            device=self.device
        )
        
        current_pos = 0
        for i, (thought_logits, _) in enumerate(continuous_thoughts):
            length = thought_logits.size(1)
            logits[i, current_pos:current_pos + length] = thought_logits
            current_pos += length
            
        return logits
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> TensorDict:
        example = self.examples[idx]
        continuous_thoughts = self._process_continuous_thoughts(example)
        return self._construct_input_sequence(example, continuous_thoughts)

class DatasetBuilder:
    @staticmethod
    def from_json(
        file_path: Union[str, Path],
        continuous_config: ContinuousThoughtConfig,
    ) -> List[ReasoningExample]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        with open(file_path) as f:
            data = json.load(f)
            
        examples: List[ReasoningExample] = []
        for item in data:
            steps = [
                ReasoningStep(text=step)
                for step in item.get("reasoning_steps", [])
            ]
            
            example = ReasoningExample(
                question=item["question"],
                answer=item["answer"],
                reasoning_steps=steps,
                continuous_config=continuous_config
            )
            examples.append(example)
            
        return examples