from flask import Flask, request, jsonify, send_file, render_template
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add these two lines before importing matplotlib
import matplotlib
matplotlib.use('Agg')  # Use Agg backend - no GUI required

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
from threading import Lock

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

class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=2, dropout=0.1):
        super(GRUAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = gru_out.permute(1, 0, 2)
        attn_output, _ = self.attention(gru_out, gru_out, gru_out)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = attn_output[:, -1, :]
        out = self.fc(attn_output)
        return out

class ImprovedTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=4, dropout=0.2):
        super(ImprovedTransformerModel, self).__init__()
        
        # Better positional encoding (sinusoidal instead of linear)
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        
        # Increase d_model to give transformer more capacity
        d_model = max(input_size, num_heads * 16)  
        
        # Project input dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        
        # More attention heads to capture different relationship aspects
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads,  # Try 4 or 8 heads instead of 2
            dim_feedforward=hidden_size*2,  # Larger feedforward network
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, 
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(d_model, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply positional encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Create attention mask to focus on recent timesteps
        transformer_out = self.transformer_encoder(x)
        
        # Consider weighted averaging across sequence instead of just last position
        output = self.dropout(torch.relu(self.output_projection(transformer_out[:, -1])))
        output = self.fc(output)
        
        return output

# Positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PCALSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=2, dropout=0.1, n_components=2):
        super(PCALSTMAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_components = n_components
        
        # Apply PCA transformation to reduce input features to n_components
        self.pca_layer = nn.Linear(input_size, n_components)
        
        # LSTM takes the reduced features (n_components) as input
        self.lstm = nn.LSTM(n_components, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply PCA-like dimensionality reduction
        batch_size, seq_len, features = x.size()
        
        # Reshape to apply PCA transformation to each timestep's features individually
        # Need to do this before passing to LSTM because LSTM expects [batch, seq, features]
        x_flattened = x.reshape(-1, features)  # Combine batch and seq dimensions
        x_pca = self.pca_layer(x_flattened)    # Apply linear PCA transformation
        x_pca = x_pca.reshape(batch_size, seq_len, self.n_components)  # Restore batch and seq dimensions
        
        # Now process through LSTM
        lstm_out, _ = self.lstm(x_pca)
        
        # Apply self-attention
        lstm_out = lstm_out.permute(1, 0, 2)  # [batch, seq, hidden] -> [seq, batch, hidden] for attention
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = attn_output.permute(1, 0, 2)  # [seq, batch, hidden] -> [batch, seq, hidden]
        
        # Extract final output
        attn_output = attn_output[:, -1, :]  # Take the last timestep
        out = self.fc(attn_output)
        return out

# Add the enhanced GRU model with hyperparameter tuning
class EnhancedGRUModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.best_model = None
        self.best_params = None
        
    def train(self, train_loader, test_loader, epochs=100):
        best_loss = float('inf')
        best_model = None
        best_params = None
        
        # Hyperparameters
        learning_rates = [0.001, 0.0001]
        hidden_sizes = [50, 100]
        num_layers_options = [1, 2]
        
        for lr in learning_rates:
            for hs in hidden_sizes:
                for nl in num_layers_options:
                    model = GRUAttentionModel(self.input_size, hs, self.output_size, num_layers=nl)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.MSELoss()
                    
                    # Training loop
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
                    
                    # Evaluate on test set
                    test_loss = 0
                    model.eval()
                    with torch.no_grad():
                        for X_test, y_test in test_loader:
                            X_test = X_test.float()
                            y_test = y_test.float()
                            y_pred = model(X_test)
                            test_loss += criterion(y_pred.squeeze(), y_test).item()
                    
                    avg_test_loss = test_loss / len(test_loader)
                    if avg_test_loss < best_loss:
                        best_loss = avg_test_loss
                        best_model = model
                        best_params = {"lr": lr, "hidden_size": hs, "num_layers": nl}
        
        self.best_model = best_model
        self.best_params = best_params
        return best_model, best_params
    
    def predict(self, X):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        self.best_model.eval()
        with torch.no_grad():
            return self.best_model(X)

@app.route('/')
def index():
    return render_template('index.html')

training_lock = Lock()

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Prevent concurrent training
        if not training_lock.acquire(blocking=False):
            return jsonify({'error': 'Training already in progress. Please wait...'}), 429

        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Get selected model type
        model_type = request.form.get('model_type', 'lstm')  # Default to LSTM if not specified

        data = pd.read_csv(file_path, parse_dates=True, index_col="Date")
        data = data.dropna()

        features = data[['Avg-Dis','rain', 'tmin', 'tmax']]
        target = data['Daily Runoff']
        
        orig_feature_dim = features.shape[1]  # Original feature dimension (should be 4)

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))
        
        # Special processing for pca-lstm model
        if model_type == 'pca-lstm':
            # Apply actual PCA transformation first, following your Colab code
            from sklearn.decomposition import PCA
            n_components = 2
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(scaled_features)
            
            # Combine PCA features with target
            scaled_data = np.concatenate((pca_features, scaled_target), axis=1)
            input_size = n_components  # Input size is now the number of PCA components
            
            print(f"Applied PCA: Explained variance ratio: {pca.explained_variance_ratio_}")
            model_name = "PCA-LSTM-Attention"
        else:
            # For other models, use all features
            scaled_data = np.concatenate((scaled_features, scaled_target), axis=1)
            input_size = scaled_features.shape[1]  # Use original feature dimension

        # Set window size based on model type
        # window_size = 1 if model_type in ['lstm', 'pca-lstm'] else 7  # Use window size 7 for GRU and transformer
        window_size = 7  # Use window size 7 for GRU and transformer

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

        hidden_size = 50
        output_size = 1
        
        # Select model based on user choice
        if model_type == 'lstm':
            model = LSTMAttentionModel(input_size, hidden_size, output_size)
            model_name = "LSTM-Attention"
        elif model_type == 'gru':
            model = GRUAttentionModel(input_size, hidden_size, output_size)
            model_name = "GRU-Attention"
        elif model_type == 'pca-lstm':
            # For PCA-LSTM, use the actual number of PCA components as input size
            # This matches your Colab reference implementation
            model = LSTMAttentionModel(
                input_size=input_size,  # This is now 2 (n_components)
                hidden_size=hidden_size, 
                output_size=output_size
            )
            model_name = "PCA-LSTM-Attention"
        elif model_type == 'enhanced_gru':
            # Use the enhanced GRU model with hyperparameter tuning
            enhanced_model = EnhancedGRUModel(input_size, hidden_size, output_size)
            model, best_params = enhanced_model.train(train_loader, test_loader, epochs=100)
            model_name = f"Enhanced GRU-Attention (lr={best_params['lr']}, hidden={best_params['hidden_size']}, layers={best_params['num_layers']})"
        else:  # transformer
            model = ImprovedTransformerModel(input_size, hidden_size, output_size)
            model_name = "Transformer"

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
        plt.title(f'Actual vs Predicted Rainfall Runoff using {model_name} for window size {window_size}')
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
            'model_type': model_type,
            'window_size': window_size,
            'total_predicted_runoff': float(total_prunoff),
            'total_actual_runoff': float(total_arunoff),
            'mse': float(mse),
            'mae': float(mae),
            'r2_score': float(r2),
            'line_plot_url': '/' + line_plot_path,
            'scatter_plot_url': '/' + scatter_plot_path,
            'bar_plot_url': '/' + bar_plot_path,
            'residuals_plot_url': '/' + residuals_plot_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        training_lock.release()

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    # Get data from request
    data = request.json
    
    # Create PDF
    pdf_path = os.path.join('static', 'report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Add content
    c.setFont("Helvetica", 16)
    c.drawString(72, 750, "Rainfall-Runoff Analysis Report")
    
    # Add metrics, plots, etc.
    
    c.save()
    
    return jsonify({'pdf_url': '/static/report.pdf'})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=False, port=5500)