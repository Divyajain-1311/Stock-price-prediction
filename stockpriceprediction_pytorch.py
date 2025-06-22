import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

# Set seed for reproducibility
torch.manual_seed(7)

# Load the dataset
df = pd.read_csv('apple_share_price.csv', usecols=[1, 2, 3, 4])  # Open, High, Low, Close
df = df.iloc[::-1]  # reverse the dataset

# Create OHLC average
OHLC_avg = df.mean(axis=1).values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
OHLC_scaled = scaler.fit_transform(OHLC_avg)

# Function to create time-series data
def create_sequences(data, step_size=1):
    X, Y = [], []
    for i in range(len(data) - step_size):
        X.append(data[i:i+step_size])
        Y.append(data[i+step_size])
    return np.array(X), np.array(Y)

# Train-test split
train_size = int(len(OHLC_scaled) * 0.75)
train_data = OHLC_scaled[:train_size]
test_data = OHLC_scaled[train_size:]

step_size = 1
X_train, y_train = create_sequences(train_data, step_size)
X_test, y_test = create_sequences(test_data, step_size)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Reshape for LSTM input: (batch, seq_len, input_size)
X_train = X_train.view(-1, step_size, 1)
X_test = X_test.view(-1, step_size, 1)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(5):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Make predictions
model.eval()
train_preds = model(X_train).detach().numpy()
test_preds = model(X_test).detach().numpy()

# Inverse transform for real values
train_preds = scaler.inverse_transform(train_preds)
y_train_real = scaler.inverse_transform(y_train.detach().numpy())
test_preds = scaler.inverse_transform(test_preds)
y_test_real = scaler.inverse_transform(y_test.detach().numpy())
OHLC_real = scaler.inverse_transform(OHLC_scaled)

# RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_real, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test_real, test_preds))
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Plotting
train_plot = np.empty_like(OHLC_real)
train_plot[:, :] = np.nan
train_plot[step_size:len(train_preds)+step_size, :] = train_preds

test_plot = np.empty_like(OHLC_real)
test_plot[:, :] = np.nan
test_plot[len(train_preds)+(step_size*2):len(OHLC_real), :] = test_preds

plt.figure(figsize=(12,6))
plt.plot(OHLC_real, label='Original Data')
plt.plot(train_plot, label='Train Predictions')
plt.plot(test_plot, label='Test Predictions')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Stock Price (OHLC avg)')
plt.title('Stock Price Prediction using PyTorch LSTM')
plt.show()
