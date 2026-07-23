"""Autoregressive sampling from a trained GPTLanguageModel."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    prompt_ids,
    max_new_tokens: int,
    block_size: int,
    device,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_id: int = 2,
):
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        if top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cum_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
        else:
            next_token = torch.multinomial(probs, 1)

        idx = torch.cat([idx, next_token], dim=1)
        if next_token.item() == eos_id:
            break

    model.train()
    return idx[0].tolist()
