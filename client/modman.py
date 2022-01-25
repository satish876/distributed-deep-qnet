import torch
import requests
from os import getpid


# Fetch Latest Model Params (StateDict)
def fetch_params(url: str):
    # Send GET request
    r = requests.get(url=url)

    # Extract data in json format
    data = r.json()

    # Check for Iteration Number (-1 Means, No model params is present on Server)
    if data['iteration'] == -1:
        return {}, False, data['iteration']
    else:
        return data['params'], True, data['iteration']
# remove send gradient method as we are not dealing with gradients in FL

# Send Trained Model Params (StateDict)


def send_model_update(url: str, params: dict, mem_size: int, steps: int, itr: int):
    body = {
        'iteration': itr,
        'model': params,
        'pid': getpid(),
        'mem_size': mem_size,
        'steps': steps
    }

    # Send POST request
    r = requests.post(url=url, json=body)

    # Extract data in json format
    data = r.json()

    return data


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
