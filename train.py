from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from neural_net import init_layer_params, full_fp, full_bp, binary_crossentopy, get_accuracy, apply_gradients, nn_architecture

# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=76)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

def fit(x, y, arch, epochs, lr, verbose=False, callbacks=None, init_seed=2):
    
    params = init_layer_params(arch, init_seed)
    
    loss_history = []
    accuracy_history = []
    
    for i in range(epochs):
        
        y_pred, tape = full_fp(x, params, arch)
        
        loss = binary_crossentopy(y_pred, y)
        loss_history.append(loss)
        accuracy = get_accuracy(y_pred, y)
        accuracy_history.append(accuracy)
        
        gradients = full_bp(y_pred, y, tape, params, arch)
        params = apply_gradients(params, gradients, arch, lr)
        
        if i % 50 == 0:
            if verbose:
                print(f"Iteration {i}: accuracy {accuracy}, loss: {loss} ")
            if callbacks is not None:
                for callback in callbacks:
                    callback(i, params, loss, accuracy)
    return params, (loss_history, accuracy_history)

params_values, history = fit(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), nn_architecture, 10000, 0.01, verbose=True, init_seed=99)

Y_test_hat, _ = full_fp(np.transpose(X_test), params_values, nn_architecture)
acc_test = get_accuracy(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f} - David".format(acc_test))

plt.plot(history[0])
plt.show()
