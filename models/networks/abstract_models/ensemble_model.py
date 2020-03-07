from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from typing import List
from abc import abstractmethod

from models.networks.abstract_models.base_model import BaseModel


class EnsembleModel:
    """
    Class that encapsulates an ensemble model. Holds a list of Tensors that are concatenated
    for the ensemble.

    Attributes
    ----------
    model
        The model that will be compiled and trained. This will be initialized with None and will remain None until the
        ensemble model is merged. Once the model is merged, this attribute will contain the merged layer. Once the model has
        been compiled, this will contain a keras Model.
    sub_models: List[keras.models.Sequential]
        The models that compose the overarching ensemble model.
    input_layers: List
        Attribute that contains all input layers of the ensemble model.
    input_channels: int
        The number of sub-models that the ensemble model should contain.
    homogeneous_models: bool
        If true, then all sub-models will have the same architecture. If false, then model architecture can vary between
        sub-models.
    """
    keras_model = None
    sub_models: List = list()
    input_layers: List = list()
    input_channels: int = None
    homogeneous_models: bool = True

    def __init__(self):
        pass

    @staticmethod
    def builder():
        """This method provides a builder for EnsembleModels."""
        return EnsembleModelBuilder()

    def add_input_layers(self, input_shape=None):
        """
        Adds input layers to the model. This is a separate function from add_layer because there are issues when adding
        input layers with add_layer and trying to merge models.

        Parameters
        ----------
        input_shape: Tuple<int, int> or List<Tuple<int, int>>
            The shape of the input layer(s) to be added. In a model w/ homogeneous sub-models, pass a single Tuple and
            an input layer will be created for each sub-model using the Tuple. In models w/ non-homogeneous sub-models,
            an input layer will be created for each tuple in the List.
        """
        if self.homogeneous_models:
            self.input_layers = [Input(shape=input_shape) for _ in range(self.input_channels)]
        else:
            if len(input_shape) != self.input_channels:
                print(f'Expected a list of {self.input_channels} tuples, but got a list of length {len(input_shape)}')
            else:
                self.input_layers = [Input(shape=s) for s in input_shape]

    def add_layer(self, layer, model_idx=None):
        """
        This method adds a layer to the ensemble model. If the model has homogeneous sub-models, then the layer will be added
        to all sub-models. If the model has non-homogeneous sub-models, then this will add a layer to a specified model.

        Parameters
        ----------
        layer: tf.Tensor or Keras Layer
            The layer that will be added to the model(s).

        model_idx: int
            The index of the sub-model the layer will be added to. If None, the layer will be added to all sub-models.
        """
        if len(self.sub_models) == 0:
            self.sub_models = [None for _ in range(self.input_channels)]

        # For adding layers after a sub-model merge
        if self.keras_model is not None:
            self.keras_model = layer(self.keras_model)

        # For adding layers to all sub-models
        elif self.homogeneous_models:
            for idx in range(self.input_channels):
                if self.sub_models[idx] is None:
                    self.sub_models[idx] = layer(self.input_layers[idx])
                else:
                    self.sub_models[idx] = layer(self.sub_models[idx])

        # For adding layers to specific sub-models in non-homogeneous ensemble
        elif model_idx is not None:
            try:
                if self.sub_models[model_idx] is None:
                    self.sub_models[model_idx] = layer(self.input_layers[model_idx])
                else:
                    self.sub_models[model_idx] = layer(self.sub_models[model_idx])
            except IndexError:
                print(
                    f'Could not find a model at index {model_idx}. This model has {self.input_channels} input channels.')
        else:
            print(f'Unable to add a layer to a non-homogeneous EnsembleModel without specifying the model index.')

    def merge_sub_models(self, func, **kwargs):
        """
        Merges sub models based on a function

        Parameters
        ----------
        func: Callable
            The function that will be invoked to perform the merge across models. This function should have a positional
            argument that takes a list of Tensors. Can optionally pass kwargs to the function.
        """
        # if self.homogeneous_models:
        #     sub_models = [Lambda(lambda x: identity(x))(self.sub_models[0]) for _ in range(self.input_channels)]
        # else:
        #     sub_models = self.sub_models
        if len(self.sub_models) == 1:
            self.keras_model = self.sub_models[0]
        else:
            self.keras_model = func([model for model in self.sub_models], **kwargs)

    def summary(self, **kwargs):
        """ Simple wrapper for Keras.model.summary"""
        return self.keras_model.summary(**kwargs)

    def compiler(self):
        """Returns a ModelCompiler that will compile the model."""
        return ModelCompiler(self)

    def compile(self):
        self.keras_model = Model(self.input_layers, self.keras_model)


class EnsembleModelBuilder:
    """
    Simple builder for EnsembleModels. It is used to make the creation of empty EnsembleModels easier.
    This builder will not be used to define the architecture of a model.

    Attributes
    ----------
    model: EnsembleModel
        The model that is being built by the builder.
    """
    model: EnsembleModel = None

    def with_input_channels(self, input_channels: int):
        if self.model is None:
            self.model = EnsembleModel()
        self.model.input_channels = input_channels
        return self

    def with_homogeneous_models(self, homogeneous_models: bool):
        if self.model is None:
            self.model = EnsembleModel()
        self.model.homogeneous_models = homogeneous_models
        return self

    def build(self):
        return self.model


class ModelCompiler:
    """
    Builder-like class used for compiling Ensemble models. Allows the use of builder-like syntax for compilation of models.

    Attributes
    ----------
    model: EnsembleModel
        The model that will be compiled
    optimizer: str
        The optimizer to use when compiling.
    loss: str
        The loss to use when compiling.
    metrics: List[str]
        List of metrics to use when compiling.
    .
    .
    .

    Methods
    -------
    compile
        Compiles the model.

    Example Usage
    -------------
    model.compiler().with_loss(<some_loss>).with_metrics(<some_metrics>).with_optimizer(<some_optimizer>).compile()
    """
    model: EnsembleModel
    optimizer: str = None
    loss: str = None
    metrics: List[str] = None
    loss_weights = None
    sample_weight_mode = None
    weighted_metrics: List = None
    target_tensors = None

    def __init__(self, model):
        self.model = model

    def with_optimizer(self, optimizer):
        self.optimizer = optimizer
        return self

    def with_loss(self, loss):
        self.loss = loss
        return self

    def with_metrics(self, metrics):
        self.metrics = metrics
        return self

    def with_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights
        return self

    def with_sample_weight_mode(self, sample_weight_mode):
        self.sample_weight_mode = sample_weight_mode
        return self

    def with_weighted_metrics(self, weighted_metrics):
        self.weighted_metrics = weighted_metrics
        return self

    def with_target_tensors(self, target_tensors):
        self.target_tensors = target_tensors
        return self

    def compile(self, **kwargs):
        self.model.compile()
        return self.model.keras_model.compile(
            self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            loss_weights=self.loss_weights,
            sample_weight_mode=self.sample_weight_mode,
            weighted_metrics=self.weighted_metrics,
            target_tensors=self.target_tensors,
            **kwargs)
