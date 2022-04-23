import torch

class FeatureExtractor(torch.nn.Module):
    def __init__(self, shape, activation):
        """ 
        Arguments:
            shape: list of number of nodes in each layer
            activation: pytorch function specifying the activation function for all the layers
        """
        super().__init__()
        self.shape = shape 
        self.layers =  []
        self.activation = activation
        # self.softmax = torch.nn.Softmax(dim=1) 

        for i in range(len(shape) - 1):
            layer = torch.nn.Linear(shape[i], shape[i + 1])
            if self.activation == torch.nn.functional.relu:
                torch.nn.init.kaiming_normal_(layer.weight)
            else:
                torch.nn.init.xavier_normal_(layer.weight)
            self.layers.append(layer)

        self.layers = torch.nn.ModuleList(self.layers)
        self.extract_feature_mode = False

    def set_extract_feature_mode(self, status: bool):
        self.extract_feature_mode = status

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        if self.extract_feature_mode:
            # return 2nd to last layer (not logits layer) 
            # represents last layer of learned features
            return x

        else:
            return self.layers[-1](x)