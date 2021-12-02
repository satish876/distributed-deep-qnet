from typing import List
import torch


def update_model(grads: dict, global_model: dict, learning_rate: float) -> dict:
    for key in global_model.keys():
        if key in grads.keys():
            global_model[key] = _apply_grads(
                global_model[key], grads[key], learning_rate)

    return global_model

@torch.no_grad()   
def RMSprop_update(params,
                    grads,
                    square_avgs,
                    weight_decay,
                    lr,
                    eps,
                    alpha):

    for i, param in enumerate(params):
        grad = torch.Tensor(grads[i])
        square_avg = square_avgs[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        avg = square_avg.sqrt().add_(eps)
        param.addcdiv_(grad, avg, value=-lr)
    return params
    

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

# Convert State Dict List to Tensor
def convert_list_to_tensor(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = torch.tensor(params[key], dtype=torch.float32)

    return params_


# Convert State Dict Tensors to List
def convert_tensor_to_list(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = params[key].tolist()

    return params_
