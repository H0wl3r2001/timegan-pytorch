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

def train_model(model, X_train, y_train, num_epochs=5, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def generate_residuals(model, train_data, window_size):
    residuals = []
    for i in range(len(train_data) - window_size):
        window_X_train = train_data[i:i + window_size].unsqueeze(-1).unsqueeze(0)  # Add batch dimension
        window_y_train = train_data[i + 1:i + window_size + 1].unsqueeze(-1).unsqueeze(0)  # Add batch dimension
        
        train_model(model, window_X_train, window_y_train, num_epochs=50, learning_rate=0.01)
        model.eval()
        
        with torch.no_grad():
            #prediction = model(window_X_train[-1].unsqueeze(0)).item()
            #residual = window_y_train[-1, -1].item() - prediction
            prediction = model(window_X_train)  # Predict next value
            prediction_value = prediction.cpu().detach().numpy().item()  # Extract scalar value
            #print(f"Prediction value for window {i}: ", prediction_value)
            actual_value = train_data[i + window_size].item()  # Corresponding actual value
            #print(f"Actual value for window {i}: ", actual_value)
            residual = actual_value - prediction_value  # Calculate residual
            residuals.append(residual)
    
    return residuals

def bootstrap_predictions_with_sliding_window(model, original_train_data, test_data, residuals, num_bootstrap=100, num_epochs=50, learning_rate=0.01):
    all_predictions = []
    current_train_data = original_train_data.clone()
    window_size = len(original_train_data)
    
    for i in range(len(test_data)):
        # Use a sliding window of fixed size equal to the original training data length
        if len(current_train_data) > window_size:
            current_train_data = current_train_data[-window_size:]
        
        X_train = current_train_data[:-1].unsqueeze(-1).unsqueeze(0)  # Add batch dimension
        y_train = current_train_data[1:].unsqueeze(-1).unsqueeze(0)  # Add batch dimension
        
        train_model(model, X_train, y_train, num_epochs=num_epochs, learning_rate=learning_rate)
        model.eval()
        
        with torch.no_grad():
            #prediction = model(X_train[-1].unsqueeze(0)).item()
            prediction = model(X_train)  # Predict next value
            prediction_value = prediction.cpu().detach().numpy().item()  # Extract scalar value
        
        bootstrapped_predictions = [prediction_value + np.random.choice(residuals) for _ in range(num_bootstrap)]
        lower_bound = np.percentile(bootstrapped_predictions, 2.5)
        upper_bound = np.percentile(bootstrapped_predictions, 97.5)
        
        all_predictions.append((prediction, lower_bound, upper_bound))
        
        # Slide the window forward by incorporating the next test data point
        current_train_data = torch.cat((current_train_data, test_data[i:i+1]), dim=0)
    
    return all_predictions


def prepare_data_tensor(df):
    return torch.tensor(df['val'].values, dtype=torch.float32)


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
    confidence_interval_diff = upper_bound_predictions - lower_bound_predictions
    
    # Calculate and return the average difference
    avg_confidence_interval_diff = np.mean(confidence_interval_diff)
    
    return avg_confidence_interval_diff