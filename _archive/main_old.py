import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

'''
We provide time-dependent sequence data of a biological process, 
aggregated by experiment ID (column "Exp"), with varying durations. 
The objective is to predict the titer measured at the final time point.

Variable Nomenclature:
X:{name}: Represents measurements of process conditions, which are variables inherent to the process.
W:{name}: Represents control conditions, which are measurements of the controlled parameters in the process.
Z:{name}: Represents control setpoints, which are operator-defined control conditions that remain constant throughout the process.
Y:{name}: Represents process attributes, which are measured at the end of the process.
'''

# TODO Task 1: Data analysis
############################

# Load the data
train_data = pd.read_csv(r"data/datahow_interview_train_data.csv")
print(train_data.head())
targets = pd.read_csv(r"data/datahow_interview_train_targets.csv")
print(targets.head())

def load_and_prepare_data(train_data, targets):
    """
    Load and prepare the data for analysis
    """
    # Merge the datasets
    merged_data = pd.merge(
        train_data, 
        targets,
        on=['Exp', 'Time[day]'],
        how='left'
    )
    
    return merged_data

def analyze_basic_statistics(data):
    """
    Calculate basic statistics for all variables
    """
    # Separate variables by type
    x_vars = [col for col in data.columns if col.startswith('X:')]
    w_vars = [col for col in data.columns if col.startswith('W:')]
    z_vars = [col for col in data.columns if col.startswith('Z:')]
    
    # Calculate statistics
    stats_x = data[x_vars].describe()
    stats_w = data[w_vars].describe()
    stats_z = data[z_vars].describe()
    
    return {
        'process_vars': stats_x,
        'control_vars': stats_w,
        'setpoint_vars': stats_z
    }

def analyze_process_patterns(data):
    """
    Analyze patterns in process variables
    """
    # Calculate key process metrics for each experiment
    metrics = pd.DataFrame()
    
    for exp in data['Exp'].unique():
        exp_data = data[data['Exp'] == exp]
        
        # Growth metrics
        metrics.loc[exp, 'Peak_VCD'] = exp_data['X:VCD'].max()
        metrics.loc[exp, 'Time_to_Peak_VCD'] = exp_data['X:VCD'].idxmax()
        metrics.loc[exp, 'Growth_Rate'] = exp_data['X:VCD'].pct_change().mean()
        
        # Metabolic metrics
        metrics.loc[exp, 'Final_Glucose'] = exp_data['X:Glc'].iloc[-1]
        metrics.loc[exp, 'Final_Glutamine'] = exp_data['X:Gln'].iloc[-1]
        metrics.loc[exp, 'Final_Ammonia'] = exp_data['X:Amm'].iloc[-1]
        metrics.loc[exp, 'Final_Lactate'] = exp_data['X:Lac'].iloc[-1]
        
        # Cell death metrics
        metrics.loc[exp, 'Final_Viability'] = (
            exp_data['X:VCD'].iloc[-1] / 
            (exp_data['X:VCD'].iloc[-1] + exp_data['X:Lysed'].iloc[-1])
        ) * 100
        
        # Final titer
        metrics.loc[exp, 'Titer'] = exp_data['Y:Titer'].iloc[-1]
    
    return metrics

def analyze_correlations(data):
    """
    Analyze correlations between variables and final titer
    """
    # Get final values for each experiment
    final_values = data.groupby('Exp').last().reset_index()
    
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

def plot_process_variables(data, exp_ids=['Exp 1', 'Exp 2', 'Exp 3']):
    """
    Plot process variables over time for selected experiments
    """
    process_vars = [col for col in data.columns if col.startswith('X:')]
    
    # Create figure with correct number of subplots
    fig, axes = plt.subplots(len(process_vars), 1, figsize=(15, 5*len(process_vars)))
    
    # Convert axes to array if there's only one subplot
    if len(process_vars) == 1:
        axes = np.array([axes])
    
    # Plot each process variable
    for i, var in enumerate(process_vars):
        for exp_id in exp_ids:
            exp_data = data[data['Exp'] == exp_id]
            axes[i].plot(exp_data['Time[day]'], exp_data[var], 
                        marker='o', label=exp_id)
        
        axes[i].set_title(f'{var} Over Time')
        axes[i].set_xlabel('Time (days)')
        axes[i].set_ylabel(var)
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()  # Add this to ensure the plot is displayed

def plot_control_performance(data, exp_ids=['Exp 1', 'Exp 2', 'Exp 3']):
    """
    Plot control variables vs setpoints
    """
    control_pairs = [
        ('W:temp', 'Z:tempStart'),
        ('W:pH', 'Z:phStart'),
        ('W:FeedGlc', 'Z:FeedRateGlc'),
        ('W:FeedGln', 'Z:FeedRateGln')
    ]
    
    fig, axes = plt.subplots(len(control_pairs), 1, figsize=(15, 5*len(control_pairs)))
    if len(control_pairs) == 1:
        axes = [axes]  # Make axes iterable if there's only one subplot
    
    for i, (w_var, z_var) in enumerate(control_pairs):
        for exp_id in exp_ids:
            exp_data = data[data['Exp'] == exp_id]
            axes[i].plot(exp_data['Time[day]'], exp_data[w_var], 
                        marker='o', label=f'{exp_id} Actual')
            axes[i].axhline(y=exp_data[z_var].iloc[0], 
                          linestyle='--', 
                          label=f'{exp_id} Setpoint')
        
        axes[i].set_title(f'{w_var} vs {z_var}')
        axes[i].set_xlabel('Time (days)')
        axes[i].set_ylabel(w_var)
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    return fig


def plot_key_relationships(data):
    """
    Plot relationships between key variables and final titer
    """
    # Get final values for each experiment
    final_values = data.groupby('Exp').last().reset_index()
    
    # Select key variables to plot
    key_vars = [
        'X:VCD', 'X:Glc', 'X:Gln', 'X:Amm', 'X:Lac', 'X:Lysed',
        'W:temp', 'W:pH', 'W:FeedGlc', 'W:FeedGln'
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, var in enumerate(key_vars):
        axes[i].scatter(final_values[var], final_values['Y:Titer'])
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Titer')
        axes[i].set_title(f'{var} vs Titer')
        
        # Add trend line
        z = np.polyfit(final_values[var], final_values['Y:Titer'], 1)
        p = np.poly1d(z)
        axes[i].plot(final_values[var], p(final_values[var]), "r--", alpha=0.8)
    
    plt.tight_layout()
    return fig

# Additional visualization: Correlation heatmap
def plot_correlation_heatmap(data):
    # Get final values for each experiment
    final_values = data.groupby('Exp').last().reset_index()
    
    # Select variables for correlation
    vars_to_correlate = [col for col in final_values.columns if col.startswith(('X:', 'W:', 'Z:', 'Y:'))]
    correlation_matrix = final_values[vars_to_correlate].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Process Variables')
    plt.tight_layout()
    plt.show()

# Additional analysis: Distribution of key variables
def plot_variable_distributions(data):
    # Get final values for each experiment
    final_values = data.groupby('Exp').last().reset_index()
    
    # Plot distributions of key variables
    key_vars = ['X:VCD', 'X:Glc', 'X:Gln', 'X:Amm', 'X:Lac', 'Y:Titer']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, var in enumerate(key_vars):
        sns.histplot(data=final_values, x=var, ax=axes[i])
        axes[i].set_title(f'Distribution of {var}')
    
    plt.tight_layout()
    plt.show()


# Function to run all analyses
def run_complete_analysis(train_data, targets):
    """
    Run all analyses and return results
    """
    # Load and prepare data
    data = load_and_prepare_data(train_data, targets)
    
    # Run analyses
    basic_stats = analyze_basic_statistics(data)
    correlations = analyze_correlations(data)
    process_patterns = analyze_process_patterns(data)
    
    # Create plots
    process_plots = plot_process_variables(data)
    control_plots = plot_control_performance(data)
    relationship_plots = plot_key_relationships(data)
    
    return {
        'statistics': basic_stats,
        'correlations': correlations,
        'process_patterns': process_patterns,
        'plots': {
            'process': process_plots,
            'control': control_plots,
            'relationships': relationship_plots
        }
    }


def display_complete_analysis(train_data, targets):
    """
    Run and display all analyses with clear formatting
    """
    # Create merged dataset first
    merged_data = load_and_prepare_data(train_data, targets)
    
    # Run the complete analysis
    results = run_complete_analysis(train_data, targets)
    
    # 1. Display basic statistics
    print("\n" + "="*50)
    print("BASIC STATISTICS")
    print("="*50)
    print("\nProcess Variables (X:):")
    print(results['statistics']['process_vars'])
    print("\nControl Variables (W:):")
    print(results['statistics']['control_vars'])
    print("\nSetpoint Variables (Z:):")
    print(results['statistics']['setpoint_vars'])
    
    # 2. Display correlations
    print("\n" + "="*50)
    print("CORRELATIONS WITH TITER")
    print("="*50)
    print("\nTop 10 correlations:")
    print(results['correlations'].head(10))
    print("\nBottom 10 correlations:")
    print(results['correlations'].tail(10))
    
    # 3. Display process patterns
    print("\n" + "="*50)
    print("PROCESS PATTERNS")
    print("="*50)
    print("\nSummary statistics for process metrics:")
    print(results['process_patterns'].describe())
    
    # 4. Show all plots
    print("\n" + "="*50)
    print("GENERATING PLOTS")
    print("="*50)
    
    # Clear any existing plots
    plt.close('all')
    
    # Process variables plot
    print("\nGenerating process variables plot...")
    plot_process_variables(merged_data)
    
    # Control performance plot
    print("\nGenerating control performance plot...")
    plot_control_performance(merged_data)
    
    # Key relationships plot
    print("\nGenerating key relationships plot...")
    plot_key_relationships(merged_data)
    
    # Correlation heatmap
    print("\nGenerating correlation heatmap...")
    plot_correlation_heatmap(merged_data)
    
    # Variable distributions
    print("\nGenerating variable distributions...")
    plot_variable_distributions(merged_data)
    
    return results



if __name__ == "__main__":
    # Load the data
    train_data = pd.read_csv(r"data/datahow_interview_train_data.csv")
    targets = pd.read_csv(r"data/datahow_interview_train_targets.csv")
    
    # Add this at the beginning of your script
    print("\nData shapes:")
    print(f"Train data shape: {train_data.shape}")
    print(f"Targets shape: {targets.shape}")
    print("\nColumns in train_data:")
    print(train_data.columns.tolist())
    
    # Create merged dataset
    merged_data = load_and_prepare_data(train_data, targets)
    
    # Run the analysis
    print("Starting complete analysis...")
    analysis_results = display_complete_analysis(train_data, targets)

    # TODO Task 2: Build data processing pipeline and model of your choice
    ############################

    # 1. Naive approach with Random Forest Regressor
    # Merge the datasets
    data = pd.merge(train_data, targets, on=['RowID', 'Exp', 'Time[day]'])

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_data, targets['Y:Titer'], test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


    # TODO Task 3: Evaluate the performance of your model
    ########################################################

    # Evaluate the model on the validation set
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    print(f'R-squared on validation set: {r2:.2f}')

    # TODO Task 4: Prepare your code so that it can generate predictions and results on a holdout test set
    ################################################################################################################


