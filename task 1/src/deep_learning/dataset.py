from typing import Dict, List, Any
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

# Виносимо мапінги на рівень констант для всього проєкту
LABEL2ID = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}
ID2LABEL = {0: "O", 1: "B-MOUNTAIN", 2: "I-MOUNTAIN"}

class NERDatasetBuilder:
    """Converts JSON data into a Hugging Face Dataset suitable for BERT."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast, label_to_id: Dict[str, int] = LABEL2ID):
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

    def build_hf_dataset(self, data: List[Dict[str, Any]]) -> Dataset:
        tokens_list = [item["tokens"] for item in data]
        labels_list = [item["labels"] for item in data]

        hf_dataset = Dataset.from_dict({
            "tokens": tokens_list,
            "ner_tags": labels_list
        })

        tokenized_dataset = hf_dataset.map(
            self._tokenize_and_align_labels,
            batched=True
        )
        return tokenized_dataset

    def _tokenize_and_align_labels(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128
        )

        labels = []
        for i, label_list in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_str = label_list[word_idx]
                    label_ids.append(self.label_to_id[label_str])
                else:
                    label_ids.append(-100)
                    
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs