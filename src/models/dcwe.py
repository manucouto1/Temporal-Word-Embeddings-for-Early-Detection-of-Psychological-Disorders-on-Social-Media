import torch
import numpy as np

from torch import nn
from transformers.models.bert import BertForMaskedLM
from torch.nn import functional as F
from scipy.sparse import dok_matrix


def cosine_similarity(V1, V2):
    dot_prod = torch.einsum(
        "abc, cba -> ab", [V1, V2.permute(*torch.arange(V2.ndim - 1, -1, -1))]
    )
    norm_1 = torch.norm(V1, dim=-1)
    norm_2 = torch.norm(V2, dim=-1)
    return dot_prod / torch.einsum(
        "bc, bc -> bc", norm_1, norm_2
    )  # Scores de similitud entre embeddings estáticos y contextualizados


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


class DCWE(nn.Module):
    def __init__(
        self,
        lambda_a,
        lambda_w,
        vocab_filter=torch.tensor([]),
        n_times=10,
        *args,
        **kwargs,
    ):
        super(DCWE, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.bert_emb_layer = self.bert.get_input_embeddings()
        print(f"Model offset_components = {n_times}")
        self.offset_components = nn.ModuleList(
            [OffsetComponent() for _ in range(n_times)]
        )
        self.lambda_a = lambda_a
        self.lambda_w = lambda_w
        self.vocab_filter = vocab_filter

    # mlm_label, reviews, masks, segs, times, vocab_filter, SA
    def forward(self, reviews, times, masks, segs):
        bert_embs = self.bert_emb_layer(reviews)

        offset_last = torch.cat(
            [
                self.offset_components[int(j.item())](bert_embs[i])
                for i, j in enumerate(F.relu(times.detach().cpu() - 1))
            ],
            dim=0,
        )
        offset_now = torch.cat(
            [
                self.offset_components[int(j.item())](bert_embs[i])
                for i, j in enumerate(times.detach().cpu())
            ],
            dim=0,
        )
        offset_last = offset_last * (
            isin(reviews, self.vocab_filter)
        ).float().unsqueeze(-1).expand(-1, -1, 768)
        offset_now = offset_now * (isin(reviews, self.vocab_filter)).float().unsqueeze(
            -1
        ).expand(-1, -1, 768)

        input_embs = bert_embs + offset_now

        output = self.bert(
            inputs_embeds=input_embs,
            attention_mask=masks,
            token_type_ids=segs,
            output_hidden_states=True,
        )

        return offset_last, offset_now, output

    def loss(self, out, labels, function):
        offset_last, offset_now, output = out

        logits = output.logits
        loss = function(logits.view(-1, self.bert.config.vocab_size), labels.view(-1))
        loss += self.lambda_a * torch.norm(offset_now, dim=-1).pow(2).mean()
        loss += (
            self.lambda_w * torch.norm(offset_now - offset_last, dim=-1).pow(2).mean()
        )
        return loss

    def generate_deltas(self, texts, input_embs, output_embs, vocab_hash_map, deltas_f):
        sim_matrix = deltas_f(input_embs, output_embs).detach().cpu().numpy()

        chunk_mat = dok_matrix(
            (sim_matrix.shape[0], len(vocab_hash_map)), dtype=np.float32
        )
        aux_t = texts.cpu().numpy()

        for post_idx, post in enumerate(aux_t):
            for token in set(post):
                if token in vocab_hash_map:
                    indices = np.where(post == token)
                    # TODO estaba sum ahora mean
                    chunk_mat[post_idx, vocab_hash_map[token]] = np.mean(
                        sim_matrix[post_idx][indices]
                    )

        return chunk_mat


class OffsetComponent(nn.Module):
    def __init__(self):
        super(OffsetComponent, self).__init__()
        self.linear_1 = nn.Linear(768, 768)
        self.linear_2 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embs):
        h = self.dropout(torch.tanh(self.linear_1(embs)))
        offset = self.linear_2(h).unsqueeze(0)
        return offset


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear_1 = nn.Linear(768, 100)
        self.linear_2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sims):
        proj_1 = self.dropout(self.linear_1(sims))
        return torch.sigmoid(self.linear_2(proj_1))
