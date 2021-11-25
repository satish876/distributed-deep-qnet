import torch
import requests


# Fetch Latest Model Params (StateDict)
def fetch_params(url: str):
    # Send GET request
    r = requests.get(url=url)

    # Extract data in json format
    data = r.json()

    # Check for Iteration Number (-1 Means, No model params is present on Server)
    if data['iteration'] == -1:
        return {}, False
    else:
        return data['params'], True


# Send Trained Model Gradients (StateDict)
def send_model_update(url: str, grads: dict):
    body = grads

    # Send POST request
    r = requests.post(url=url, data=body)

    # Extract data in json format
    data = r.json()

    return data


# Convert State Dict List to Tensor
def convert_list_to_tensor(params: dict) -> dict:
    for key in params.keys():
        params[key] = torch.tensor(params[key], dtype=torch.float32)

    return params


# Convert State Dict Tensors to List
def convert_tensor_to_list(params: dict) -> dict:
    for key in params.keys():
        params[key] = params[key].tolist()

    return params
