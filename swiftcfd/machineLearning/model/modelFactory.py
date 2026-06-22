from swiftcfd.machineLearning.model.multilayerPerceptron import MultiLayerPerceptron as mlp
from swiftcfd.machineLearning.model.recurrentNeuralNetwork import RecurrentNeuralNetwork as rnn
from swiftcfd.machineLearning.model.longShortTermMemory import LongShortTermMemory as lstm
from swiftcfd.machineLearning.model.transformer import Transformer as transformer

def create_model(model_type, input_variables, output_variables, equation_type='heat',
                 input_size=7, hidden_size=256, output_size=5, num_layers=5, dropout=0.1):

    # model parameter, constructor arguments
    model_kwargs = {
        "input_variables":  input_variables,
        "output_variables": output_variables,
        "equation_type":    equation_type,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "num_layers": num_layers,
        "dropout": dropout,
    }

    if model_type == 'mlp':
        return mlp(**model_kwargs)
    elif model_type == 'rnn':
        return rnn(**model_kwargs)
    elif model_type == 'lstm':
        return lstm(**model_kwargs)
    elif model_type == 'transformer':
        return transformer(**model_kwargs)
    else:
        raise Exception('Unknown machine learning model: ' + model_type)
