from tensorflow.keras import applications as ka
from .__version__ import __version__


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', "tensorflow")
    layers = kwargs.get('layers', "layers")
    models = kwargs.get('models', "models")
    utils = kwargs.get('utils', "utils")
    return backend, layers, models, utils
