import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class RNNPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1, model='rnn'):
        super(RNNPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.module = self.get_rnn_module(model)(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.module(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
    def get_rnn_module(self, model):
        if model == 'rnn':
            return nn.RNN
        elif model == 'lstm':
            return nn.LSTM
        elif model == 'gru':
            return nn.GRU
        else:
            raise ValueError(f"Unsupported model type: {model}")

def train_model(model, X_train, y_train, num_epochs=100, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #if (epoch+1) % 10 == 0:
        #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def predict_with_confidence(model, X, alpha=0.05):
    model.eval()
    with torch.no_grad():
        predictions = model(X).cpu().detach().numpy()
        
    errors = []
    for i in range(100):
        noise = torch.normal(0, 1, size=X.size()).to(X.device)
        noisy_predictions = model(X + noise).cpu().detach().numpy()
        errors.append(noisy_predictions - predictions)
    
    errors = np.array(errors)
    lower_bound = np.percentile(errors, alpha/2*100, axis=0)
    upper_bound = np.percentile(errors, (1 - alpha/2)*100, axis=0)
    
    lower_bound_predictions = predictions + lower_bound
    upper_bound_predictions = predictions + upper_bound
    
    return predictions, lower_bound, upper_bound, lower_bound_predictions, upper_bound_predictions

def predict_next_n_with_avg_confidence_interval(rnn_model, current_data, n, alpha=0.05):
    # Prepare the input tensor for the next n predictions
    last_idx = current_data['idx'].iloc[-1]
    next_indices = np.arange(last_idx + 1, last_idx + 1 + n)
    next_indices_tensor = torch.FloatTensor(next_indices).unsqueeze(-1).unsqueeze(1)  # Shape: (batch_size, seq_length=1, input_size=1)

    # Get predictions and confidence intervals
    predictions, lower_bound, upper_bound, lower_bound_predictions, upper_bound_predictions = predict_with_confidence(rnn_model, next_indices_tensor, alpha)
    
    # Calculate the difference between upper and lower bounds for each forecasted value
    confidence_interval_diff = upper_bound - lower_bound
    
    # Calculate and return the average difference
    avg_confidence_interval_diff = np.mean(confidence_interval_diff)
    
    return avg_confidence_interval_diff