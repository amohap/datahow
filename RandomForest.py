import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(train_data_path, train_targets_path):
    """Load and preprocess the data"""
    process_data = pd.read_csv(train_data_path)
    target_data = pd.read_csv(train_targets_path)
    
    # Get final titer values for each experiment
    final_titers = target_data.drop(["RowID", "Time[day]"], axis=1)

    return process_data, final_titers

def create_basic_statistics(df):
    """ Create basic statistics (mean, std, min , perc, max)"""
    # Identify variable groups
    x_vars = [col for col in df.columns if col.startswith('X:')]
    w_vars = [col for col in df.columns if col.startswith('W:')]
    z_vars = [col for col in df.columns if col.startswith('Z:')]
    
    # Calculate summary statistics for each group
    summary_stats = {}
    for var_group, vars in [('Process (X)', x_vars), ('Control (W)', w_vars), ('Setpoint (Z)', z_vars)]:
        stats = df[vars].describe()
        summary_stats[var_group] = stats
    
    return summary_stats

def plot_variable_distributions(train_data_path, train_targets_path):
    process_data = pd.read_csv(train_data_path)
    target_data = pd.read_csv(train_targets_path)

    merged_data = pd.merge(
            process_data, 
            target_data,
            on=['Exp', 'Time[day]'],
            how='left'
        )

    # Get final values for each experiment
    final_values = merged_data.groupby('Exp').last().reset_index()
        
    # Plot distributions of key variables
    key_vars = ['X:VCD', 'X:Glc', 'X:Gln', 'X:Amm', 'X:Lac', 'Y:Titer']
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
        
    for i, var in enumerate(key_vars):
        sns.histplot(data=final_values, x=var, ax=axes[i])
        axes[i].set_title(f'Distribution of {var}')
        
    plt.tight_layout()
    plt.show()

def analyze_correlations(train_data_path, train_targets_path, plot=True):
    """ Analyze correlations between variables and final titer"""
    process_data = pd.read_csv(train_data_path)
    target_data = pd.read_csv(train_targets_path)

    merged_data = pd.merge(
        process_data, 
        target_data,
        on=['Exp', 'Time[day]'],
        how='left'
    )
    
    # Get final values for each experiment
    final_values = merged_data.groupby('Exp').last().reset_index()

    if plot:
        # Select variables for correlation
        vars_to_correlate = [col for col in final_values.columns if col.startswith(('X:', 'W:', 'Z:', 'Y:'))]
        correlation_matrix = final_values[vars_to_correlate].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Heatmap of Process Variables')
        plt.tight_layout()
        plt.show()
    
    # Calculate correlations with titer
    x_vars = [col for col in final_values.columns if col.startswith('X:')]
    w_vars = [col for col in final_values.columns if col.startswith('W:')]
    z_vars = [col for col in final_values.columns if col.startswith('Z:')]
    
    correlations = pd.DataFrame()
    
    for vars_list, var_type in zip([x_vars, w_vars, z_vars], ['Process', 'Control', 'Setpoint']):
        for var in vars_list:
            correlation = pearsonr(final_values[var].values, final_values['Y:Titer'].values)
            correlations.loc[var, 'Correlation'] = correlation[0]
            correlations.loc[var, 'P-value'] = correlation[1]
            correlations.loc[var, 'Type'] = var_type
    
    return correlations.sort_values('Correlation', ascending=False)

def engineer_features(process_data):
    """
    Simple feature engineering for bioprocess data:
    X variables: mean, std, min, max
    W variables: mean only
    Z variables: initial values only
    """
    grouped = process_data.groupby('Exp')
    engineered_features = {}
    
    # Process variables (X) - Basic statistics
    x_vars = [col for col in process_data.columns if col.startswith('X:')]
    for var in x_vars:
        engineered_features[f'{var}_mean'] = grouped[var].mean()
        engineered_features[f'{var}_std'] = grouped[var].std()
        engineered_features[f'{var}_min'] = grouped[var].min()
        engineered_features[f'{var}_max'] = grouped[var].max()
    
    # Control variables (W) - Mean only
    w_vars = [col for col in process_data.columns if col.startswith('W:')]
    for var in w_vars:
        engineered_features[f'{var}_mean'] = grouped[var].apply(lambda x: x[x != 0].mean())

    # Setpoint variables (Z) - Initial values only
    z_vars = [col for col in process_data.columns if col.startswith('Z:')]
    for var in z_vars:
        engineered_features[f'{var}_initial'] = grouped[var].first()
    
    # Create DataFrame
    feature_df = pd.DataFrame(engineered_features)
    
    return feature_df

def engineer_features_baseline(process_data):
    """ Baseline feature engineering using only mean values of X variables."""
    grouped = process_data.groupby('Exp')
    engineered_features = {}
    
    # Process variables (X) - Mean values only
    x_vars = [col for col in process_data.columns if col.startswith('X:')]
    for var in x_vars:
        engineered_features[f'{var}_mean'] = grouped[var].mean()
    
    # Create DataFrame
    feature_df = pd.DataFrame(engineered_features)
    
    return feature_df

def train_evaluate_model(X, y, selected_features):
    """Train and evaluate the model using updated metrics"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=0.2, random_state=5
    )
    
    # Scale features
    scaler=None
    
    # Train model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=500, 
        criterion='squared_error',
        max_depth=None, # let the tree grow fully
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, r2, rmse, feature_importance, scaler

def evaluate_on_test_data(model, scaler, test_data_path, test_targets_path, selected_features, baseline):
    """Evaluate the trained model on test data with fixed target values"""
    # Load test data
    test_process_data = pd.read_csv(test_data_path)
    test_targets = pd.read_csv(test_targets_path)
    
    # Engineer features for test data
    if baseline:
        test_features = engineer_features_baseline(test_process_data)
    else:
        test_features = engineer_features(test_process_data)
    
    # Get target values
    test_targets = test_targets['Y:Titer'].values
    
    
    # Make predictions
    test_predictions = model.predict(test_features[selected_features])
    
    return test_predictions, test_targets, test_features

def visualize_model_performance(y_true, y_pred, title="Model Performance"):
    """Create visualization for model performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Scatter plot of predicted vs actual values
    ax = axes[0,0]
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Perfect Prediction')
    ax.set_xlabel('Actual Titer')
    ax.set_ylabel('Predicted Titer')
    ax.set_title('Predicted vs Actual Titer')
    ax.legend()
    
    # Histogram of residuals
    residuals = y_pred - y_true
    ax = axes[0,1]
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title('Distribution of Residuals')
    ax.set_xlabel('Residual Value')
    
    # Residuals vs predicted values
    ax = axes[1,0]
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Titer')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predicted Values')
    
    # QQ plot of residuals
    ax = axes[1,1]
    
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    return fig

def print_model_metrics(y_true, y_pred, set_name="Test"):
    """Print comprehensive model evaluation metrics using updated sklearn metrics"""
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
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def visualize_predictions(y_true, y_pred, title="Model Performance"):
    """Create visualization of model predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 1. Regression plot with actual vs predicted values
    ax = axes[0,0]
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'r--', label='Perfect Prediction')
    ax.set_xlabel('Target Titer')
    ax.set_ylabel('Predicted Titer')
    ax.set_title('Predicted vs Target Titer')
    ax.legend()
    
    # 2. Prediction error plot
    residuals = y_pred - y_true
    ax = axes[0,1]
    sns.scatterplot(x=range(len(y_pred)), y=residuals, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Errors')
    
    # 3. Error distribution
    ax = axes[1,0]
    sns.histplot(residuals, kde=True, ax=ax)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_title('Distribution of Prediction Errors')
    ax.set_xlabel('Error')
    
    # 4. Q-Q Plot
    ax = axes[1,1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot of Prediction Errors')
    
    plt.suptitle(title, y=1.02, size=16)
    plt.tight_layout()
    return fig

def visualize_feature_importance(data):
    """Create visualization of feature importance"""
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=data.head(10),
        x='Importance',
        y='Feature'
    )
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.show()


def run_pipeline(train_data_path, train_targets_path, test_data_path, test_targets_path, baseline=True):
    """Run the complete analysis pipeline"""
    # Load and preprocess training data
    process_data, final_titers = load_and_preprocess_data(train_data_path, train_targets_path)

    # Perform data analysis
    summary_stats = create_basic_statistics(process_data)
    print("DATA ANALYSIS")
    print("\nProcess Variables (X:):")
    print(summary_stats['Process (X)'])
    print("\nControl Variables (W:):")
    print(summary_stats['Control (W)'])
    print("\nSetpoint Variables (Z:):")
    print(summary_stats['Setpoint (Z)'])

    print("CORRELATIONS WITH TITER")
    correlations = analyze_correlations(train_data_path, train_targets_path)
    print("\nTop 10 correlations:")
    print(correlations.head(10))
    print("\nBottom 10 correlations:")
    print(correlations.tail(10))

    plot_variable_distributions(train_data_path, train_targets_path)

    # Engineer features
    if baseline:
        feature_df = engineer_features_baseline(process_data)
    else:
        feature_df = engineer_features(process_data)
    
    # Select features
    if baseline:
        selector = SelectKBest(score_func=mutual_info_regression, k=6)
    else:
        selector = SelectKBest(score_func=mutual_info_regression, k=20) #there are 41 in total

    selector.fit(feature_df, final_titers['Y:Titer'])

    # Put the selected features into a list
    selected_features = feature_df.columns[selector.get_support()].tolist()
    
    # Train and evaluate model
    model, r2, rmse, feature_importance, scaler = train_evaluate_model(
        feature_df, final_titers['Y:Titer'], selected_features
    )
    
    # Evaluate on test data with correct targets
    test_predictions, test_targets, test_features = evaluate_on_test_data(
        model, scaler, test_data_path, test_targets_path, selected_features, baseline)
    
    # Print training metrics
    train_predictions = model.predict(feature_df[selected_features])
    train_metrics = print_model_metrics(
        final_titers['Y:Titer'], 
        train_predictions, 
        "Training"
    )
    
    # Print test metrics
    test_metrics = print_model_metrics(test_targets, test_predictions, "Test")
    
    # Create detailed predictions analysis
    prediction_analysis = pd.DataFrame({
        'Experiment': [f'Test Exp {i+1}' for i in range(len(test_predictions))],
        'Target_Titer': test_targets,
        'Predicted_Titer': test_predictions,
        'Absolute_Error': np.abs(test_predictions - test_targets),
        'Percentage_Error': np.abs((test_predictions - test_targets) / test_targets * 100)
    }).sort_values('Absolute_Error', ascending=False)
    
    print("\nDetailed Prediction Analysis:")
    print(prediction_analysis)
    
    print("\nPrediction Error Summary:")
    print(f"Mean Absolute Error: {prediction_analysis['Absolute_Error'].mean():.2f}")
    print(f"Median Absolute Error: {prediction_analysis['Absolute_Error'].median():.2f}")
    print(f"Mean Percentage Error: {prediction_analysis['Percentage_Error'].mean():.2f}%")
    print(f"Median Percentage Error: {prediction_analysis['Percentage_Error'].median():.2f}%")
    

    # Create visualizations
    train_viz = visualize_predictions(
        final_titers['Y:Titer'].values,
        train_predictions,
        "Training Set Performance"
    )
    
    test_viz = visualize_predictions(
        test_targets,
        test_predictions,
        "Test Set Performance"
    )

    # Print feature importance
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Create feature importance plot
    visualize_feature_importance(feature_importance)

    return {
        'model': model,
        'feature_importance': feature_importance,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'prediction_analysis': prediction_analysis,
        'selected_features': selected_features,
        'visualizations': {
            'train': train_viz,
            'test': test_viz
        }
    }


# Main execution
if __name__ == "__main__":
    # Run the analysis
    results = run_pipeline(
        'data/datahow_interview_train_data.csv',
        'data/datahow_interview_train_targets.csv',
        'data/datahow_interview_test_data.csv',
        'data/datahow_interview_test_targets-TRUE.csv',
        baseline=False
    )

    # Display visualizations
    plt.show()