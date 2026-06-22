import argparse as ap

class CommandLineArgumentParser:
    def __init__(self):
        # create parser
        self.parser = ap.ArgumentParser(description='A CFD solver for quick 2D equation prototyping.')

        # add arguments
        self.parser.add_argument('-i', '--input', help = 'input file for simulation',
                                                  required = False)
        self.parser.add_argument('-t', '--train', help = 'train ML model based on available datasets',
                                                  action = 'store_true',
                                                  required = False,
                                                  default = False)
        self.parser.add_argument('-iv', '--input-variables', help = 'input variables for training',
                                                  required = False)
        self.parser.add_argument('-ov', '--output-variables', help = 'output variables for inference',
                                                  required = False)
        self.parser.add_argument('-m', '--model', help = 'ML model to use for training and inference',
                                                  choices = ['mlp', 'rnn', 'lstm', 'transformer'],
                                                  required = False)
        self.parser.add_argument('--epochs',     help = 'number of training epochs (default: 200)',
                                                  type = int,
                                                  required = False,
                                                  default = 200)
        self.parser.add_argument('--batch-size', help = 'mini-batch size for training (default: 256)',
                                                  type = int,
                                                  required = False,
                                                  default = 256)
        self.parser.add_argument('--lr',         help = 'learning rate for Adam optimizer (default: 1e-4)',
                                                  type = float,
                                                  required = False,
                                                  default = 1e-4)
        self.parser.add_argument('--patience',      help = 'early-stopping patience in epochs (default: 300)',
                                                    type = int,
                                                    required = False,
                                                    default = 300)
        self.parser.add_argument('--equation-type', help = 'PDE type for physics loss: heat or ns (default: heat)',
                                                    choices = ['heat', 'ns'],
                                                    required = False,
                                                    default = 'heat')
        self.parser.add_argument('--seed',          help = 'random seed for reproducible training (default: 0)',
                                                    type = int,
                                                    required = False,
                                                    default = 0)
        self.parser.add_argument('--hidden-size',   help = 'hidden layer width (default: 256)',
                                                    type = int,
                                                    required = False,
                                                    default = 256)
        self.parser.add_argument('--num-layers',    help = 'number of hidden layers (default: 5)',
                                                    type = int,
                                                    required = False,
                                                    default = 5)
        self.parser.add_argument('--dropout',       help = 'dropout probability (default: 0.1)',
                                                    type = float,
                                                    required = False,
                                                    default = 0.1)
        self.parser.add_argument('--weight-pde',    help = 'weight of the PDE/physics loss term (default: 1.0)',
                                                    type = float,
                                                    required = False,
                                                    default = 1.0)
        self.parser.add_argument('--weight-data',   help = 'weight of the data loss term (default: 1.0)',
                                                    type = float,
                                                    required = False,
                                                    default = 1.0)
        self.parser.add_argument('--output-dir',    help = 'directory for saved model/norm artifacts (default: output)',
                                                    type = str,
                                                    required = False,
                                                    default = 'output')

        # parse arguments from command line arguments
        self.arguments = self.__parse()
    
        # perform check that all required parameters are available to continue
        self.__check_arguments()

    def __parse(self):
        return self.parser.parse_args()

    def __check_arguments(self):
        if not self.arguments.train and self.arguments.input is None:
            raise Exception('Input file is required for simulation.')

        if self.arguments.train and self.arguments.model is None:
            raise Exception('ML model is required for training.')

        if self.arguments.train and self.arguments.input_variables is None:
            raise Exception('Input variables are required for training.')

        if self.arguments.train and self.arguments.output_variables is None:
            raise Exception('Output variables are required for training.')