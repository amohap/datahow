import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Try to configure GPU, fallback to CPU if not available
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU found. Using CPU instead.")

def prepare_process_data(data):
    """Prepare process data by extracting and organizing relevant features"""
    # X variables (cell culture variables)
    x_vars = [col for col in data.columns if col.startswith('X:')]
    
    # W variables (control variables)
    w_vars = [col for col in data.columns if col.startswith('W:')]
    
    # Z variables (experimental conditions)
    z_vars = [col for col in data.columns if col.startswith('Z:')]
    
    # Add Time[day] as a feature
    feature_cols = x_vars + w_vars + z_vars + ['Time[day]']
    
    return feature_cols

def prepare_sequences(process_data, target_data=None, sequence_length=5, is_test=False):
    """Prepare sequences for LSTM model"""
    sequences = []
    targets = []
    experiment_ids = []
    
    # Get feature columns
    feature_cols = prepare_process_data(process_data)
    
    # Get all unique experiments
    experiments = process_data['Exp'].unique()
    
    for exp in experiments:
        exp_data = process_data[process_data['Exp'] == exp]
        
        # Get target if available (training data)
        if not is_test and target_data is not None:
            exp_target = target_data[target_data['Exp'] == exp]['Y:Titer'].iloc[-1]
        else:
            exp_target = None
            
        # Get process data including time
        data = exp_data[feature_cols].values
        
        # Create sequences
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:(i + sequence_length)])
            if not is_test:
                targets.append(exp_target)
            experiment_ids.append(exp)
    
    return (np.array(sequences), 
            np.array(targets) if not is_test else None, 
            experiment_ids)

def create_lstm_model(input_shape, learning_rate=0.0005):
    """Create LSTM model"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True,
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros',
             kernel_regularizer=tf.keras.regularizers.l2(1e-6)),
        Dropout(0.2),
        LSTM(32, return_sequences=True,
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros',
             kernel_regularizer=tf.keras.regularizers.l2(1e-6)),
        Dropout(0.2),
        LSTM(16, return_sequences=False,
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             bias_initializer='zeros',
             kernel_regularizer=tf.keras.regularizers.l2(1e-6)),
        Dense(16, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=tf.keras.regularizers.l2(1e-6)),
        Dense(1)
    ])
    
    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_and_predict(train_data_path, train_targets_path, test_data_path, sequence_length=5):
    """Train LSTM model and make predictions"""
    # Load data
    train_data = pd.read_csv(train_data_path)
    train_targets = pd.read_csv(train_targets_path)
    test_data = pd.read_csv(test_data_path)
    
    print(f"\nUsing fixed sequence length: {sequence_length} days")
    
    # Prepare sequences
    X_train, y_train, train_exp_ids = prepare_sequences(
        train_data, train_targets, sequence_length, is_test=False
    )
    X_test, _, test_exp_ids = prepare_sequences(
        test_data, None, sequence_length, is_test=True
    )
    
    # Handle any infinite values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    n_samples, n_timesteps, n_features = X_train.shape
    
    # Reshape and scale training data
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Scale targets
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    
    # Scale test data
    n_samples_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
    
    # Create and train model
    model = create_lstm_model((sequence_length, n_features))
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        min_delta=1e-4
    )
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_split=0.2,
        epochs=150,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    train_pred_scaled = model.predict(X_train_scaled, batch_size=16)
    test_pred_scaled = model.predict(X_test_scaled, batch_size=16)
    
    # Inverse transform predictions
    train_pred = y_scaler.inverse_transform(train_pred_scaled)
    test_pred = y_scaler.inverse_transform(test_pred_scaled)
    
    # Handle any NaN values in predictions
    train_pred = np.nan_to_num(train_pred)
    test_pred = np.nan_to_num(test_pred)
    
    # Calculate training metrics
    print_model_metrics(y_train, train_pred, "Training")
    
    # Create predictions DataFrame for test data
    test_predictions = pd.DataFrame({
        'Experiment': list(set(test_exp_ids)),
        'Predicted_Titer': [test_pred[np.array(test_exp_ids) == exp].mean() 
                           for exp in set(test_exp_ids)]
    }).sort_values('Experiment')
    
    # Plot training history
    plot_training_history(history)
    
    return {
        'model': model,
        'test_predictions': test_predictions,
        'history': history,
        'scalers': (scaler, y_scaler)
    }

def plot_training_history(history):
    """Plot LSTM training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_test_evaluation(eval_df):
    """Plot test predictions evaluation"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Predictions vs Actual
    ax = axes[0,0]
    ax.scatter(eval_df['Y:Titer'], eval_df['Predicted_Titer'], alpha=0.5)
    ax.plot([eval_df['Y:Titer'].min(), eval_df['Y:Titer'].max()],
            [eval_df['Y:Titer'].min(), eval_df['Y:Titer'].max()],
            'r--', label='Perfect Prediction')
    ax.set_xlabel('Actual Titer')
    ax.set_ylabel('Predicted Titer')
    ax.set_title('Predicted vs Actual Titer')
    ax.legend()
    
    # Error Distribution
    ax = axes[0,1]
    sns.histplot(eval_df['Percentage_Error'], kde=True, ax=ax)
    ax.set_title('Distribution of Prediction Errors')
    ax.set_xlabel('Percentage Error')
    
    # Predictions by Experiment
    ax = axes[1,0]
    eval_df.plot(x='Experiment', y=['Predicted_Titer', 'Y:Titer'], 
                kind='bar', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title('Predictions vs Actual by Experiment')
    
    # Error by Experiment
    ax = axes[1,1]
    sns.barplot(data=eval_df, x='Experiment', y='Percentage_Error', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title('Prediction Error by Experiment')
    
    plt.tight_layout()
    plt.show()

def print_model_metrics(y_true, y_pred, set_name="Test"):
    """Print comprehensive model evaluation metrics"""
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Handle MAPE calculation with zero values
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                             y_true[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    print(f"\n{set_name} Set Metrics:")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MAPE: {mape:.2f}%")

def evaluate_test_predictions(predictions_df, test_targets_path):
    """Evaluate predictions against actual test targets"""
    # Load actual test targets
    test_targets = pd.read_csv(test_targets_path)
    
    # Merge predictions with actual values
    evaluation_df = predictions_df.merge(
        test_targets[['Exp', 'Y:Titer']].groupby('Exp').last(),
        left_on='Experiment',
        right_index=True
    )
    
    # Calculate errors
    evaluation_df['Absolute_Error'] = abs(evaluation_df['Predicted_Titer'] - 
                                        evaluation_df['Y:Titer'])
    evaluation_df['Percentage_Error'] = (abs(evaluation_df['Predicted_Titer'] - 
                                           evaluation_df['Y:Titer']) / 
                                       evaluation_df['Y:Titer'] * 100)
    
    # Print metrics
    print("\nTest Set Evaluation:")
    print_model_metrics(
        evaluation_df['Y:Titer'],
        evaluation_df['Predicted_Titer'],
        "Test"
    )
    
    # Create visualizations
    plot_test_evaluation(evaluation_df)
    
    return evaluation_df


# Train the model
results = train_and_predict(
    'data/datahow_interview_train_data.csv',
    'data/datahow_interview_train_targets.csv',
    'data/datahow_interview_test_data.csv',
    sequence_length=5  # Adjust this value to use more or less history
)

# Evaluate predictions
evaluation = evaluate_test_predictions(
    results['test_predictions'],
    'data/datahow_interview_test_targets-TRUE.csv'
)