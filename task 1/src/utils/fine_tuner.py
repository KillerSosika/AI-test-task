import transformers
from typing import Dict, Optional
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)

from src.training.base_trainer import BaseTrainer
from src.deep_learning.dataset import LABEL2ID, ID2LABEL

class BERTFineTuner(BaseTrainer):
    """Wrapper for fine-tuning a BERT model for Token Classification."""

    def __init__(self, model_name: str, output_dir: str = "models/finetuned"):
        self.model_name = model_name
        self.output_dir = output_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def train(self, train_data: Dataset, val_data: Optional[Dataset] = None, **kwargs) -> None:
        epochs = kwargs.get("epochs", 3)
        batch_size = kwargs.get("batch_size", 16)
        learning_rate = kwargs.get("learning_rate", 2e-5)

        training_kwargs = {
            "output_dir": self.output_dir,
            "save_strategy": "epoch",
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": epochs,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "load_best_model_at_end": True if val_data else False,
            "save_total_limit": 2
        }

        hf_version = tuple(map(int, transformers.__version__.split(".")[:2]))
        eval_strategy_val = "epoch" if val_data else "no"

        if hf_version >= (4, 41):
            training_kwargs["eval_strategy"] = eval_strategy_val
        else:
            training_kwargs["evaluation_strategy"] = eval_strategy_val

        training_args = TrainingArguments(**training_kwargs)

        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_data,
            "eval_dataset": val_data,
            "data_collator": self.data_collator,
        }

        try:
            trainer = Trainer(**trainer_kwargs, processing_class=self.tokenizer)
        except TypeError:
            trainer = Trainer(**trainer_kwargs, tokenizer=self.tokenizer)

        print(f"Starting training for {epochs} epochs...")
        trainer.train()
        self.save(self.output_dir)

    def evaluate(self, val_data: Dataset) -> Dict[str, float]:
        pass

    def save(self, path: str) -> None:
        print(f"Saving final model to {path}...")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)