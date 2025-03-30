import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load dataset
data = pd.read_csv('D:/frza/rainfall_runoff/KR-8037-16649-f.csv', parse_dates=True, index_col="Date")
data = data.dropna()

# Extract features and target
features = data[['Avg-Dis','rain', 'tmin', 'tmax']]
target = data['Daily Runoff']

# Scale data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))
scaled_data = np.concatenate((scaled_features, scaled_target), axis=1)

# Define window size
window_size = 7

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(torch.tensor(data[i:i+window_size, :-1]))
        y.append(torch.tensor(data[i+window_size, -1]))
    return X, y

X, y = create_sequences(scaled_data, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(pad_sequence(X_train, batch_first=True), torch.stack(y_train))
test_dataset = TensorDataset(pad_sequence(X_test, batch_first=True), torch.stack(y_test))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=2, dropout=0.1):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = attn_output[:, -1, :]
        out = self.fc(attn_output)
        return out

input_size = X_train[0].shape[-1]
model = LSTMAttentionModel(input_size, hidden_size=50, output_size=1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
train_losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.float(), y_batch.float()
        optimizer.zero_grad()
        loss = criterion(model(X_batch).squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]}')

test_loss, predictions, actuals = 0, [], []
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.float(), y_test.float()
        y_pred = model(X_test)
        predictions.extend(y_pred.squeeze().tolist())
        actuals.extend(y_test.tolist())
        test_loss += criterion(y_pred.squeeze(), y_test).item()

print(f'Test MSE: {test_loss / len(test_loader)}')
print(f'Test MAE: {mean_absolute_error(actuals, predictions)}')
print(f'Test R^2 score: {r2_score(actuals, predictions)}')

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

plt.figure(figsize=(10, 6), dpi=300)
sns.lineplot(x=range(len(actuals)), y=actuals.ravel(), label='Actual')
sns.lineplot(x=range(len(predictions)), y=predictions.ravel(), label='Predicted')
plt.xlabel('Time')
plt.ylabel('Runoff (m³/s)')
plt.title('Actual vs Predicted Rainfall Runoff')
plt.legend()
plt.show()

# Calculate total runoff
total_prunoff = np.sum(predictions)
total_arunoff = np.sum(actuals)
print(f'Total Predicted runoff: {total_prunoff} m³/s')
print(f'Total actual runoff: {total_arunoff} m³/s')

# Create a DataFrame for Seaborn
data = pd.DataFrame({
    'Time': range(len(actuals)),
    'Actual': actuals.ravel(),
    'Predicted': predictions.ravel()
})

# Line plot
plt.figure(figsize=(10, 6), dpi=300)
sns.lineplot(data=data, x='Time', y='Actual', label='Actual')
sns.lineplot(data=data, x='Time', y='Predicted', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Runoff (m³/s)')
plt.title('Actual vs Predicted Rainfall Runoff using Attentional LSTM for window size 7')
plt.legend()
plt.savefig("Krishna-River-LSTM-Att-Runoff-W7LSTM-LinePlot", dpi=300)
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Time', y='Actual', label='Actual', color='blue', alpha=0.5)
sns.scatterplot(data=data, x='Time', y='Predicted', label='Predicted', color='red', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Runoff (m³/s)')
plt.title('Actual vs Predicted Rainfall Runoff (Scatter Plot)')
plt.legend()
plt.savefig("Krishna-River-LSTM-Att-Runoff-W7LSTM-ScatterPlot", dpi=300)
plt.show()

# Bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=data.melt(id_vars='Time', var_name='Type', value_name='Discharge'), x='Time', y='Discharge', hue='Type')
plt.xlabel('Time')
plt.ylabel('Runoff (m³/s)')
plt.title('Actual vs Predicted Rainfall Runoff (Bar Plot)')
plt.legend()
plt.savefig("Krishna-River-LSTM-Att-Runoff-W7LSTM-BarPlot", dpi=300)
plt.show()

# Residuals plot
data['Residuals'] = data['Actual'] - data['Predicted']
plt.figure(figsize=(10, 6), dpi=300)
sns.residplot(x=data['Time'], y=data['Residuals'], lowess=True, color='red')
plt.xlabel('Time')
plt.ylabel('Residuals (m³/s)')
plt.title('Residuals Plot')
plt.savefig("Krishna-River-LSTM-Att-Runoff-W7LSTM-ResidualsPlot", dpi=300)
plt.show()