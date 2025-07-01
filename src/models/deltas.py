import torch
from torch.nn import functional as F


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(m.log(), p, reduction="none").sum(-1)
        + F.kl_div(m.log(), q, reduction="none").sum(-1)
    )


def wasserstein_distance(u_values, v_values):
    def compute_wasserstein(u, v):
        u_sorter = torch.argsort(u, dim=-1)
        v_sorter = torch.argsort(v, dim=-1)

        all_values = torch.cat([u, v], dim=-1)
        all_values, _ = torch.sort(all_values, dim=-1)

        deltas = torch.diff(all_values, dim=-1)

        u_cdf = torch.searchsorted(
            torch.gather(u, -1, u_sorter), all_values[..., :-1], right=True
        )
        v_cdf = torch.searchsorted(
            torch.gather(v, -1, v_sorter), all_values[..., :-1], right=True
        )

        return torch.sum(torch.abs(u_cdf - v_cdf) * deltas, dim=-1)

    distances = torch.stack(
        [
            compute_wasserstein(u_values[:, i, :], v_values[:, i, :])
            for i in range(u_values.shape[1])
        ],
        dim=1,
    )

    return distances


def pairwise_distance(input_embs, output_embs, p=2):
    diff = input_embs - output_embs
    distances = torch.norm(diff, p=p, dim=-1)

    return distances


DISTANCES = {
    "cosine": lambda input_embs, output_embs: (
        1 - F.cosine_similarity(input_embs, output_embs, dim=-1)
    )
    / 2,
    "euclidean": lambda input_embs, output_embs: pairwise_distance(
        input_embs, output_embs, p=2
    ),
    "manhattan": lambda input_embs, output_embs: pairwise_distance(
        input_embs, output_embs, p=1
    ),
    "jensen_shannon": lambda input_embs, output_embs: jensen_shannon_divergence(
        input_embs, output_embs
    ),
    "wasserstein": lambda input_embs, output_embs: wasserstein_distance(
        input_embs, output_embs
    ),
    "chebyshev": lambda input_embs, output_embs: torch.max(
        torch.abs(input_embs - output_embs), dim=-1
    ).values,
    "minkowski": lambda input_embs, output_embs: pairwise_distance(
        input_embs, output_embs, p=3
    ).pow(1 / 3),
}


def f_generate_deltas(input_embs, output_embs, distance_metric="cosine"):
    if distance_metric in DISTANCES:
        sim_matrix = DISTANCES[distance_metric](input_embs, output_embs)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    return sim_matrix
