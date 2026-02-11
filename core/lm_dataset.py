import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, text_field="text", max_length=512):
        super().__init__()
        self.data_path = str(data_path)
        self.path = Path(self.data_path)
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length

        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id.")
        if self.bos_token_id is None or self.eos_token_id is None:
            raise ValueError("Tokenizer must define bos_token_id and eos_token_id.")

        self.is_jsonl = self.path.suffix == ".jsonl"
        if self.is_jsonl:
            self.offsets = self._build_offsets()
            if not self.offsets:
                raise ValueError(
                    f"No usable rows found in {self.data_path}. Check text_field={self.text_field!r}."
                )
        else:
            text = self.path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"Empty text corpus in {self.data_path}")
            self.samples = [line for line in text.splitlines() if line.strip()]
            if not self.samples:
                self.samples = [text]

    def _build_offsets(self):
        offsets = []
        with self.path.open("r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                row = line.strip()
                if not row:
                    continue
                obj = json.loads(row)
                value = obj.get(self.text_field, "")
                if isinstance(value, str) and value:
                    offsets.append(offset)
                elif value:
                    offsets.append(offset)
        return offsets

    def __len__(self):
        if self.is_jsonl:
            return len(self.offsets)
        return len(self.samples)

    def _read_sample(self, index):
        if self.is_jsonl:
            with self.path.open("r", encoding="utf-8") as f:
                f.seek(self.offsets[index])
                row = json.loads(f.readline())
            value = row.get(self.text_field, "")
            return value if isinstance(value, str) else str(value)
        return self.samples[index]

    def __getitem__(self, index):
        text = self._read_sample(index)
        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids

        tokens = [self.bos_token_id] + token_ids + [self.eos_token_id]
        length = len(tokens)
        if length > self.max_length:
            tokens = tokens[: self.max_length]
            length = self.max_length

        input_ids = tokens + [self.pad_token_id] * (self.max_length - length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = input_ids.clone()
        labels[input_ids == self.pad_token_id] = -100

        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:length] = 1

        return input_ids, labels, attention_mask
