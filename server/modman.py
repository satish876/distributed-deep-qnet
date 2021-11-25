from typing import List
import torch


def update_model(grads: dict, global_model: dict, learning_rate: float) -> dict:
    for key in global_model.keys():
        if key in grads.keys():
            global_model[key] = _apply_grads(
                global_model[key], grads[key], learning_rate)

    return global_model


def _apply_grads(param: list, grad: list, lr: float):

    # Convert To Torch Tensors
    grad_ = torch.tensor(grad, dtype=torch.float32)
    param_ = torch.tensor(param, dtype=torch.float32)

    # Apply Gradient Update
    accu_grads = torch.zeros(param_.size())
    accu_grads.add_(grad_, alpha=-lr)
    param_.add_(accu_grads)

    # Convert to List and Return
    return param_.tolist()
