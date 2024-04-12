import torch

class UnivariateRNN(torch.nn.Module):
    r"""A general RNN model for time-series prediction
    """
    def __init__(self, args):
        super(UnivariateRNN, self).__init__()
        self.model_type = args['model_type']

        self.input_size = args['in_dim']
        self.hidden_size = 1 #args['h_dim']
        self.output_size = args['out_dim']
        self.num_layers = args['n_layers']
        self.dropout = args['dropout']
        self.bidirectional = args['bidirectional']

        self.padding_value = args['padding_value']
        self.max_seq_len = args['max_seq_len']
    
        self.rnn_module = self._get_rnn_module(self.model_type)
        
        self.rnn_layer = self.rnn_module(
            input_size=self.input_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        
        self.linear_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size
        )

    def _get_rnn_module(self, model_type):
        if model_type == "rnn":
            return torch.nn.RNN
        elif model_type == "lstm":
            return torch.nn.LSTM
        elif model_type == "gru":
            return torch.nn.GRU
        
    def forward(self, X, T):
        X, _ = self.rnn_layer(X)
        X = self.linear_layer(X)
        return X