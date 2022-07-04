import numpy as np

nn_architecture = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

# Initialize weights and biases
def init_layer_params(arch, seed=99):
    
    np.random.seed(seed)
    
    params = {}
    
    for idx, layer in enumerate(arch):
        layer_idx = idx + 1
        
        params[f"W{layer_idx}"] = np.random.randn(
            layer["output_dim"], layer["input_dim"]
        ) * 0.1
        
        params[f"B{layer_idx}"] = np.random.randn(
            layer["output_dim"], 1
        ) * 0.1
    
    return params

# Activation functions and their derivatives

def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0,z)

def sigmoid_delta(da,z):
    sig = sigmoid(z)
    return da * sig * (1 - sig)

def relu_delta(da, z):
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0
    return dz

# Forward propogation

def single_layer_fp(a_prev, w_curr, b_curr, activation="sigmoid"):

    activation_func = globals().get(activation)
    if not activation_func:
        raise ValueError(f"Activation function: {activation} not yet implemented")
    
    z = np.dot(w_curr, a_prev) + b_curr
    a = activation_func(z)
    
    return a, z

def full_fp(x, params, arch):

    tape = {}
    
    a_curr = x
    
    for idx, layer in enumerate(arch):
        layer_idx = idx + 1
        
        w_curr = params[f"W{layer_idx}"]
        b_curr = params[f"B{layer_idx}"]
        
        activation_func = layer["activation"]
        
        a, z = single_layer_fp(a_curr, w_curr, b_curr, activation_func)
        
        tape[f"A{idx}"] = a_curr
        tape[f"Z{layer_idx}"] = z
        
        a_curr = a

    return a_curr, tape

# Loss function, and accuracy function

def binary_crossentopy(y_pred, y):

    m = y_pred.shape[1]

    loss = -1 / m * (np.dot(y, np.log(y_pred).T) + np.dot(1 - y, np.log(1 - y_pred).T))
    return np.squeeze(loss)

def probs_to_class(probs):
    classes = np.copy(probs)
    classes[probs > 0.5] = 1
    classes[probs <= 0.5] = 0
    return classes

def get_accuracy(y_pred, y):

    y_pred = probs_to_class(y_pred)
    return (y_pred == y).all(axis=0).mean()

# Backwards propogartion (Shudders) whyyyyy

def single_layer_bp(da_curr, w_curr, b_curr, z_curr, a_prev, activation="relu"):
    
    m = a_prev.shape[1]
    
    activation_func_delta = globals().get(f"{activation}_delta")
    if not activation_func_delta:
        raise ValueError(f"Activation function delta of: {activation} not yet implemented")
    
    dz = activation_func_delta(da_curr, z_curr)
    dw = np.dot(dz, a_prev.T) / m
    db = np.sum(dz, axis=1, keepdims=True) / m
    da_prev = np.dot(w_curr.T, dz)
    
    return dw, db, da_prev

def full_bp(y_pred, y, tape, params, arch):

    gradients = {}
    
    m = y.shape[1]
    y = y.reshape(y_pred.shape)
    
    da_prev = - (np.divide(y, y_pred) - np.divide(1 - y, 1 - y_pred))
    
    for prev_layer_idx, layer in reversed(list(enumerate(arch))):
        layer_idx = prev_layer_idx + 1
        layer_activation_func = layer["activation"]
        
        da_curr = da_prev
        
        a_prev = tape[f"A{prev_layer_idx}"]
        z_curr = tape[f"Z{layer_idx}"]
        
        w_curr = params[f"W{layer_idx}"]
        b_curr = params[f"B{layer_idx}"]
        
        dw, db, da_prev = single_layer_bp(da_curr, w_curr, b_curr, z_curr, a_prev, layer_activation_func)
        
        gradients[f"dW{layer_idx}"] = dw
        gradients[f"dB{layer_idx}"] = db
    
    return gradients

def apply_gradients(params, gradients, arch, lr):

    for layer_idx, layer in enumerate(arch, 1):
        params[f"W{layer_idx}"] -= lr * gradients[f"dW{layer_idx}"]
        params[f"B{layer_idx}"] -= lr * gradients[f"dB{layer_idx}"]
    
    return params
