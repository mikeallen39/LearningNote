import numpy as np

def batch_norm(x, gemma, beta, eps):
    x_mean = np.mean(x, axis=(0,1), keepdims=True)
    x_var = np.var(x, axis=(0,1), keepdims=True)
    x_norm = (x - x_mean) / (np.sqrt(x_var + eps))
    return gemma * x_norm + beta

def layer_norm(x, gemma, beta, eps):
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - x_mean) / (np.sqrt(x_var + eps))
    return gemma * x_norm + beta

def rms_norm(x, gemma, eps):
    x_rms = np.mean(x ** 2, axis=-1, keepdims=True)
    return gemma * x / (np.sqrt(x_rms + eps))