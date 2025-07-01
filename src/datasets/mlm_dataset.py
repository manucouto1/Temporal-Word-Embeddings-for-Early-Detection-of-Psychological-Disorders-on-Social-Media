from torch.utils.data import Dataset
import torch


class MlmChunkDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        texts,
        _excluded_token_ids,
        mlm_probability=0.15,
        max_length=512,
    ):
        self.examples = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.mlm_probability = mlm_probability
        self.tokenizer = tokenizer
        self._excluded_token_ids = _excluded_token_ids

    def __len__(self):
        return self.examples["input_ids"].size(0)

    def __getitem__(self, idx):
        input_ids = self.examples["input_ids"][idx].clone()
        attention_mask = self.examples["attention_mask"][idx]

        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )

        if self._excluded_token_ids:
            excluded_mask = torch.isin(
                input_ids, torch.tensor(list(self._excluded_token_ids))
            )
            probability_matrix.masked_fill_(excluded_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        input_ids[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100  # ignore for loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
