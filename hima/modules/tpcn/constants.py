from hima.modules.tpcn.utils import Softmax, Sigmoid, Tanh, ReLU
EPS = 1e-24

ACTIVATION_FUNCS = {
    "softmax": Softmax(),
    "tanh": Tanh(),
    "sigmoid": Sigmoid(),
    "relu": ReLU(),
}