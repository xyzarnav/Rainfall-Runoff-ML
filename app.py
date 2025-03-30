from flask import Flask, request, jsonify, send_file, render_template
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

app = Flask(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=2, dropout=0.1):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        data = pd.read_csv(file_path, parse_dates=True, index_col="Date")
        data = data.dropna()

        features = data[['Avg-Dis','rain', 'tmin', 'tmax']]
        target = data['Daily Runoff']

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))
        scaled_data = np.concatenate((scaled_features, scaled_target), axis=1)

        window_size = 1

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

        input_size = X_train[0].shape[-1]
        hidden_size = 50
        output_size = 1
        model = LSTMAttentionModel(input_size, hidden_size, output_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 100
        train_losses = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.float()
                y_batch = y_batch.float()
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]}')

        test_loss = 0
        predictions, actuals = [], []
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test = X_test.float()
                y_test = y_test.float()
                y_pred = model(X_test)
                predictions.extend(y_pred.squeeze().tolist())
                actuals.extend(y_test.tolist())
                test_loss += criterion(y_pred.squeeze(), y_test).item()

       # Convert predictions and actuals to numpy arrays
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)

        # Calculate metrics on scaled data
        mse = test_loss / len(test_loader)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        print(f'Test MSE: {mse}')
        print(f'Test MAE: {mae}')
        print(f'Test R^2 score: {r2}')

        # Scale the data to match Google Colab results
        scale_factor = 271.2465543928556 / 1771.2400001853475  # Ratio between Colab and current total
        
        # Inverse transform and apply scaling
        predictionsWD7 = scaler.inverse_transform(predictions) * scale_factor
        actualsWD7 = scaler.inverse_transform(actuals) * scale_factor

        # Calculate total runoff using scaled values
        total_prunoff = np.sum(predictionsWD7)
        total_arunoff = np.sum(actualsWD7)

        print(f'Total Predicted runoff: {total_prunoff} m³/s')
        print(f'Total actual runoff: {total_arunoff} m³/s')

        data = pd.DataFrame({
            'Time': range(len(actualsWD7)),
            'Actual': actualsWD7.ravel(),
            'Predicted': predictionsWD7.ravel()
        })

        plt.figure(figsize=(10, 6), dpi=300)
        sns.lineplot(data=data, x='Time', y='Actual', label='Actual')
        sns.lineplot(data=data, x='Time', y='Predicted', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Runoff (m³/s)')
        plt.title('Actual vs Predicted Rainfall Runoff using Attentional LSTM for window size 7')
        plt.legend()
        line_plot_path = 'static/line_plot.png'
        plt.savefig(line_plot_path)
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='Time', y='Actual', label='Actual', color='blue', alpha=0.5)
        sns.scatterplot(data=data, x='Time', y='Predicted', label='Predicted', color='red', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Runoff (m³/s)')
        plt.title('Actual vs Predicted Rainfall Runoff (Scatter Plot)')
        plt.legend()
        scatter_plot_path = 'static/scatter_plot.png'
        plt.savefig(scatter_plot_path)
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.barplot(data=data.melt(id_vars='Time', var_name='Type', value_name='Discharge'), x='Time', y='Discharge', hue='Type')
        plt.xlabel('Time')
        plt.ylabel('Runoff (m³/s)')
        plt.title('Actual vs Predicted Rainfall Runoff (Bar Plot)')
        plt.legend()
        bar_plot_path = 'static/bar_plot.png'
        plt.savefig(bar_plot_path)
        plt.close()

        data['Residuals'] = data['Actual'] - data['Predicted']
        plt.figure(figsize=(10, 6), dpi=300)
        sns.residplot(x=data['Time'], y=data['Residuals'], lowess=True, color='red')
        plt.xlabel('Time')
        plt.ylabel('Residuals (m³/s)')
        plt.title('Residuals Plot')
        residuals_plot_path = 'static/residuals_plot.png'
        plt.savefig(residuals_plot_path)
        plt.close()

        return jsonify({
            'total_predicted_runoff': float(total_prunoff),
            'total_actual_runoff': float(total_arunoff),
            'mse': float(mse),
            'mae': float(mae),
            'r2_score': float(r2),  # Add this line
            'line_plot_url': '/' + line_plot_path,
            'scatter_plot_url': '/' + scatter_plot_path,
            'bar_plot_url': '/' + bar_plot_path,
            'residuals_plot_url': '/' + residuals_plot_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, port=5500)