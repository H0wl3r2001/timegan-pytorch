# -*- coding: UTF-8 -*-
import torch
import numpy as np

from models.utils import timegan_generator
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from metrics.arima import prepare_data2, prepare_data
from metrics.rnn_confidence import bootstrap_predictions_with_sliding_window, generate_residuals, prepare_data_tensor

class EmbeddingNetwork(torch.nn.Module):
    """The embedding network (encoder) for TimeGAN
    """
    def __init__(self, args):
        super(EmbeddingNetwork, self).__init__()
        self.feature_dim = args.feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Embedder Architecture
        self.emb_rnn = torch.nn.GRU(
            input_size=self.feature_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.emb_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.emb_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.emb_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.emb_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, X, T):
        """Forward pass for embedding features from original space into latent space
        Args:
            - X: input time-series features (B x S x F)
            - T: input temporal information (B)
        Returns:
            - H: latent space embeddings (B x S x H)
        """
        # Dynamic RNN input for ignoring paddings
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=X, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 128 x 100 x 71
        H_o, H_t = self.emb_rnn(X_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )
        
        # 128 x 100 x 10
        logits = self.emb_linear(H_o)
        # 128 x 100 x 10
        H = self.emb_sigmoid(logits)
        return H

class RecoveryNetwork(torch.nn.Module):
    """The recovery network (decoder) for TimeGAN
    """
    def __init__(self, args):
        super(RecoveryNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.feature_dim = args.feature_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Recovery Architecture
        self.rec_rnn = torch.nn.GRU(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.rec_linear = torch.nn.Linear(self.hidden_dim, self.feature_dim)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.rec_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.rec_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for the recovering features from latent space to original space
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - X_tilde: recovered data (B x S x F)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 128 x 100 x 10
        H_o, H_t = self.rec_rnn(H_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100 x 71
        X_tilde = self.rec_linear(H_o)
        return X_tilde

class SupervisorNetwork(torch.nn.Module):
    """The Supervisor network (decoder) for TimeGAN
    """
    def __init__(self, args):
        super(SupervisorNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Supervisor Architecture
        self.sup_rnn = torch.nn.GRU(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers-1,
            batch_first=True
        )
        self.sup_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sup_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.sup_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.sup_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for the supervisor for predicting next step
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - H_hat: predicted next step data (B x S x E)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 128 x 100 x 10
        H_o, H_t = self.sup_rnn(H_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100 x 10
        logits = self.sup_linear(H_o)
        # 128 x 100 x 10
        H_hat = self.sup_sigmoid(logits)
        return H_hat

class GeneratorNetwork(torch.nn.Module):
    """The generator network (encoder) for TimeGAN
    """
    def __init__(self, args):
        super(GeneratorNetwork, self).__init__()
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Generator Architecture
        self.gen_rnn = torch.nn.GRU(
            input_size=self.Z_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.gen_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gen_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.gen_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.gen_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, Z, T):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information
        Returns:
            - H: embeddings (B x S x E)
        """
        # Dynamic RNN input for ignoring paddings
        Z_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=Z, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 128 x 100 x 71
        H_o, H_t = self.gen_rnn(Z_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100 x 10
        logits = self.gen_linear(H_o)
        # B x S
        H = self.gen_sigmoid(logits)
        return H

class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN
    """
    def __init__(self, args):
        super(DiscriminatorNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Discriminator Architecture
        self.dis_rnn = torch.nn.GRU(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.dis_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.dis_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information
        Returns:
            - logits: predicted logits (B x S x 1)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 128 x 100 x 10
        H_o, H_t = self.dis_rnn(H_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100
        logits = self.dis_linear(H_o).squeeze(-1)
        return logits

class TimeGAN(torch.nn.Module):
    """Implementation of TimeGAN (Yoon et al., 2019) using PyTorch
    Reference:
    - https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html
    - https://github.com/jsyoon0823/TimeGAN
    """
    def __init__(self, args, T2, base_data, test_data, aciw, model=None):
        super(TimeGAN, self).__init__()
        self.device = args.device
        self.feature_dim = args.feature_dim
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.T2 = T2
        self.args = args
        self.model = model
        self.base_data = base_data
        self.test_data = test_data
        self.aciw = aciw

        self.embedder = EmbeddingNetwork(args)
        self.recovery = RecoveryNetwork(args)
        self.generator = GeneratorNetwork(args)
        self.supervisor = SupervisorNetwork(args)
        self.discriminator = DiscriminatorNetwork(args)

    def _recovery_forward(self, X, T):
        """The embedding network forward pass and the embedder network loss
        Args:
            - X: the original input features
            - T: the temporal information
        Returns:
            - E_loss: the reconstruction loss
            - X_tilde: the reconstructed features
        """
        # Forward Pass
        H = self.embedder(X, T)
        X_tilde = self.recovery(H, T)

        # For Joint training
        H_hat_supervise = self.supervisor(H, T)
        G_loss_S = torch.nn.functional.mse_loss(
            H_hat_supervise[:,:-1,:], 
            H[:,1:,:]
        ) # Teacher forcing next output

        # Reconstruction Loss
        E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X)
        E_loss0 = 10 * torch.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1 * G_loss_S
        return E_loss, E_loss0, E_loss_T0

    def _supervisor_forward(self, X, T):
        """The supervisor training forward pass
        Args:
            - X: the original feature input
        Returns:
            - S_loss: the supervisor's loss
        """
        # Supervision Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Supervised loss
        S_loss = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output
        return S_loss

    def _discriminator_forward(self, X, T, Z, gamma=1):
        """The discriminator forward pass and adversarial loss
        Args:
            - X: the input features
            - T: the temporal information
            - Z: the input noise
        Returns:
            - D_loss: the adversarial loss
        """
        # Real
        H = self.embedder(X, T).detach()
        
        # Generator
        E_hat = self.generator(Z, T).detach()
        H_hat = self.supervisor(E_hat, T).detach()

        # Forward Pass
        Y_real = self.discriminator(H, T)            # Encoded original data
        Y_fake = self.discriminator(H_hat, T)        # Output of generator + supervisor
        Y_fake_e = self.discriminator(E_hat, T)      # Output of generator

        D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
        D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))

        with torch.no_grad():
            Z1 = torch.rand((len(T), self.args.max_seq_len, self.args.Z_dim))
            X2 = self._inference(Z1, T)
            X2 = X2.cpu().detach().numpy()
            X2 = prepare_data2(X2)

        base_data = prepare_data2(self.base_data)

        avr_diff = mean_squared_error(X2['val'], base_data['val'][:len(X2['val'])])

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        return D_loss, avr_diff

    def _generator_forward(self, X, T, Z, gamma=1):
        """The generator forward pass
        Args:
            - X: the original feature input
            - T: the temporal information
            - Z: the noise for generator input
        Returns:
            - G_loss: the generator's loss
        """
        # Supervisor Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Generator Forward Pass
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)

        # Generator Loss
        # 1. Adversarial loss
        Y_fake = self.discriminator(H_hat, T)        # Output of supervisor
        Y_fake_e = self.discriminator(E_hat, T)      # Output of generator

        G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

        # 2. Supervised loss
        G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output

        # 3. Two Momments
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

        G_loss_V = G_loss_V1 + G_loss_V2

        # 4. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V

        return G_loss
    
    def _generator_forward_2nd_arima(self, X, T, Z, gamma=1, a=50):
        """The generator forward pass. 2nd phase where the output of the predictor is used as input to the generator loss function
        Args:
            - X: the original feature input
            - X2: the synthetic data generated to condition the predictor
            - T: the temporal information
            - Z: the noise for generator input
        Returns:
            - G_loss: the generator's loss
            - average_conf_int: the average difference between the values of the 95% confidence interval of the ARIMA model
        """
        # Supervisor Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Generator Forward Pass
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)

        # Generator Loss
        # 1. Adversarial loss
        Y_fake = self.discriminator(H_hat, T)        # Output of supervisor
        Y_fake_e = self.discriminator(E_hat, T)      # Output of generator

        G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

        # 2. Supervised loss
        G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output

        # 3. Two Momments
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

        G_loss_V = G_loss_V1 + G_loss_V2

        with torch.no_grad():
            Z1 = torch.rand((len(self.T2), self.args.max_seq_len, self.args.Z_dim))
            X2 = self._inference(Z1, self.T2)
            X2 = X2.cpu().detach().numpy()
            X2 = np.concatenate((np.array(self.base_data), X2[np.array(self.base_data).shape[0]:]))
            X2 = prepare_data2(X2)

        #TODO calculate the average difference between values of the 95% confidence interval and add that to the generator loss function
        # 4. ARIMA model and confidence interval loss calculation
            model = ARIMA(X2['val'].values, order=self.model)
            fitted_model = model.fit()
            #TODO: get forecast on training data and generate array of differences. In case it is not normal, one step-ahed for cycle
            #example 100 times, select one random element from the array and add to the prediction (prediction intervals residuals bootsraping method)
            forecast= fitted_model.get_forecast(len(self.T2)//3)
            #TODO: USE CWC METRIC
            conf_int = forecast.conf_int(alpha=0.05)

            average_conf_int = np.mean(conf_int[:,1] - conf_int[:,0])

            metric = torch.abs(torch.log(torch.tensor(average_conf_int/self.aciw, dtype=G_loss_U.dtype)))

        #print("G_l_u type: ", G_loss_U.dtype)
        #print("G_l_u_e type: ", G_loss_U_e.dtype)
        #print("G_l_s type: ", G_loss_S.dtype)
        #print("G_l_v type: ", G_loss_V.dtype)
        #print("metric type: ", metric.dtype)
        # 5. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V + metric

        return G_loss, average_conf_int, metric
    
    def _generator_forward_2nd_rnn(self, X, T, Z, gamma=1, a=1):
        """The generator forward pass. 2nd phase where the output of the predictor is used as input to the generator loss function
        Args:
            - X: the original feature input
            - X2: the synthetic data generated to condition the predictor
            - T: the temporal information
            - Z: the noise for generator input
        Returns:
            - G_loss: the generator's loss
            - average_conf_int: the average difference between the values of the 95% confidence interval of the rnn model
        """
        # Supervisor Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Generator Forward Pass
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)

        # Generator Loss
        # 1. Adversarial loss
        Y_fake = self.discriminator(H_hat, T)        # Output of supervisor
        Y_fake_e = self.discriminator(E_hat, T)      # Output of generator

        G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

        # 2. Supervised loss
        G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output

        # 3. Two Momments
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

        G_loss_V = G_loss_V1 + G_loss_V2

        with torch.no_grad():
            Z1 = torch.rand((len(self.T2), self.args.max_seq_len, self.args.Z_dim))
            X2 = self._inference(Z1, self.T2)
            X2 = X2.cpu().detach().numpy()
            X2 = np.concatenate((np.array(self.base_data), X2[np.array(self.base_data).shape[0]:]))
            X2 = prepare_data2(X2)
            _, test_data = prepare_data(self.base_data, self.test_data)
            test_data = test_data[len(test_data) // 2:] # Now I have training data with a third of it being synthetic (train = 75%, test = 25%) 
        
            #TODO calculate the average difference between values of the 95% confidence interval and add that to the generator loss function
            # 4. Rnn model and confidence interval loss calculation

            train_data_T = prepare_data_tensor(X2)
            test_data_T  = prepare_data_tensor(test_data)
            #rnn_model = RNNPredictor(input_size=1, hidden_size=50, num_layers=1, output_size=1, model='rnn')

        #print("Generating residuals\n")
        residuals = generate_residuals(self.model, train_data_T, len(self.base_data)) #Residuals only in the part of the synthetic data

        #print("Generating confidence intrv\n")
        predictions_with_confidence_intervals = bootstrap_predictions_with_sliding_window(self.model, train_data_T, test_data_T, residuals)

        differences = [upper - lower for _, lower, upper in predictions_with_confidence_intervals]

        average_conf_int = np.mean(differences)
        metric = torch.abs(torch.log(torch.tensor(average_conf_int/self.aciw, dtype=G_loss_U.dtype)))
            # 5. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V + a * metric

        return G_loss, average_conf_int

    def _inference(self, Z, T):
        """Inference for generating synthetic data
        Args:
            - Z: the input noise
            - T: the temporal information
        Returns:
            - X_hat: the generated data
        """
        # Generator Forward Pass
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)
        return X_hat

    def forward(self, X, T, Z, obj, gamma=1):
        """
        Args:
            - X: the input features (B, H, F)
            - T: the temporal information (B)
            - Z: the sampled noise (B, H, Z)
            - obj: the network to be trained (`autoencoder`, `supervisor`, `generator`, `discriminator`)
            - gamma: loss hyperparameter
        Returns:
            - loss: The loss for the forward pass
            - X_hat: The generated data
        """
        if obj != "inference":
            if X is None:
                raise ValueError("`X` should be given")

            X = torch.FloatTensor(X)
            X = X.to(self.device)

        if Z is not None:
            Z = torch.FloatTensor(Z)
            Z = Z.to(self.device)

        if obj == "autoencoder":
            # Embedder & Recovery
            loss = self._recovery_forward(X, T)

        elif obj == "supervisor":
            # Supervisor
            loss = self._supervisor_forward(X, T)

        elif obj == "generator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Generator
            loss = self._generator_forward(X, T, Z)

        elif obj == "generator_2nd_phase_arima":
            if Z is None:
                raise ValueError("`Z` is not given")
            
            #generate synthetic data to condition model
            #X_hat = self._inference(Z, self.T2)
            #X_hat = X_hat.cpu().detach().numpy()

            #couple orignal data with the remainder coming from the generated data, making it into data that can be used to condition the generator

            #X_hat = np.concatenate((np.array(self.base_data), X_hat[np.array(self.base_data).shape[0]:]))

            # Generator
            loss = self._generator_forward_2nd_arima(X, T, Z)

        elif obj == "discriminator":
            if Z is None:
                raise ValueError("`Z` is not given")
            
            # Discriminator
            loss = self._discriminator_forward(X, T, Z)
            
            return loss

        elif obj == "generator_2nd_phase_rnn":
            if Z is None:
                raise ValueError("`Z` is not given")
            
            #generate synthetic data to condition model
            #X_hat = self._inference(Z, self.T2)
            #X_hat = X_hat.cpu().detach().numpy()

            #couple orignal data with the remainder coming from the generated data, making it into data that can be used to condition the generator

            #X_hat = np.concatenate((np.array(self.base_data), X_hat[np.array(self.base_data).shape[0]:]))

            # Generator
            loss = self._generator_forward_2nd_rnn(X, T, Z)        

        elif obj == "inference":

            X_hat = self._inference(Z, T)
            X_hat = X_hat.cpu().detach()

            return X_hat

        else: raise ValueError("`obj` should be either `autoencoder`, `supervisor`, `generator`, or `discriminator`")

        return loss
