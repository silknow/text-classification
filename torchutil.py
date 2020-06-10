"""
Utility functions for using with pytorch
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def output_to_multiclass(out, dim=1):
    """Converts output of an output layer to a class (id) via softmax
    and argmax."""
    out = F.softmax(out, dim=dim)
    out = torch.argmax(out, dim=dim)
    return out


def output_to_multiclass_pair(out, dim=1):
    """Converts output of an output layer to a class (id) via softmax
    and argmax. Returns (class, score)."""
    out = F.softmax(out, dim=dim)

    out = torch.max(out, dim=dim)
    score = out.values.item()
    class_id = out.indices.item()

    return (class_id, score)


def output_to_multilabel(out, dim=None):
    """Converts output of an output layer to a class (id) via softmax
    and argmax."""
    out = torch.sigmoid(out)
    out = (out > 0.5)
    return out


def predict(model, data_iter):
    model.eval()
    y_prd = []
    with torch.no_grad():
        for x, y in tqdm(data_iter):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # forward
            output = model(x)

            output = output_to_multiclass(output, dim=1)
            output = output.item()
            y_prd.append(output)

    y_prd = np.array(y_prd)
    return y_prd


def predict_sko(model, data_iter, mode):
    model.eval()
    y_prd = []
    with torch.no_grad():
        for x, y in tqdm(data_iter):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # forward
            output = model(x)

            # is hidden included
            if type(output) == tuple:
                output, _ = output

            if mode == 'multiclass':
                output = output_to_multiclass(output, dim=1)
                output = output.item()
                y_prd.append(output)
            elif mode == 'multilabel':
                output = output_to_multilabel(output, dim=1)
                y_prd.append(output.cpu())
            else:
                raise ValueError(f'Invalid mode: {mode}')

    if mode == 'multiclass':
        y_prd = np.array(y_prd)
    elif mode == 'multilabel':
        y_prd = torch.cat(y_prd, dim=0).numpy()
    return y_prd


def predict_simple(model, data_iter):
    model.eval()
    y_prd = []
    with torch.no_grad():
        for x, y in tqdm(data_iter):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # forward
            output = model(x)

            # is hidden included
            if type(output) == tuple:
                output, _ = output

            output = output_to_multiclass_pair(output, dim=1)
            y_prd.append(output)

    return y_prd


def generate_hidden_vectors(model, data_iter):
    model.eval()
    hidden_vectors = []
    with torch.no_grad():
        for x, y in tqdm(data_iter):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # forward
            _, hidden = model(x)
            hidden_vectors.append(hidden)

    hidden_vectors = [h.cpu().numpy() for h in hidden_vectors]

    return hidden_vectors
