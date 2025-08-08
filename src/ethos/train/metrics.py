import math
import os

import torch as th
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


@th.inference_mode()
def estimate_loss(
    model: PreTrainedModel, ctx, loaders: list[tuple[str, DataLoader]], eval_iters: int
) -> dict:
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1
    
    out = {}
    for split, dataloader in loaders:
        # Get device from model parameters instead of model.device
        device = next(model.parameters()).device
        losses = th.empty(eval_iters, device=device)
        for i, (X, Y) in zip(range(eval_iters), dataloader):
            # Move tensors to device
            Y = Y.to(device, non_blocking=True)
            if isinstance(X, tuple):
                X = (X[0].to(device, non_blocking=True), X[1].to(device, non_blocking=True))
            else:
                X = X.to(device, non_blocking=True)
            
            with ctx:
                if isinstance(X, tuple):
                    output = model(input_ids=X[0], decoder_input_ids=X[1], labels=Y)
                else:
                    output = model(input_ids=X, labels=Y)
                loss = output.loss
            losses[i] = loss.item()

        out[f"loss/{split}"] = losses.mean().item()
    return out
