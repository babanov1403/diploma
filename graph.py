import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import ast

# Set style for all plots
plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")

def parse_data(filename):
    f = open(filename, 'r')
    content = f.read()
    data_dict = ast.literal_eval(content)
    parsed_data = {}
    list_length = None
    for key, value in data_dict.items():
        float_key = float(key)
        parsed_data[float_key] = np.array(value, dtype=float)
        
        # Check consistent list lengths
        if list_length is None:
            list_length = len(value)
    return parsed_data

def visualize_data(data_dict, idx):
    """Create plots from the parsed data"""
    if not data_dict:
        return
    
    # Prepare data for plotting
    keys = data_dict.keys()
    values = np.array([data_dict[k] for k in keys])
    num_columns = values.shape[1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(figsize=(14, 14))

    to_graph = []
    for llist in values:
        to_graph.append(llist[idx])
    # Plot each column of data
    axes.plot(keys, to_graph, linestyle='-', 
            label=f'Parameter {idx+1}')
    axes.set_ylabel(f'Value {idx+1}')
    axes.grid(True, alpha=0.3)
    axes.legend()
    
    axes.set_xlabel('Key Value')
    fig.suptitle('Parameter Evolution Across Keys', y=1.02)
    plt.tight_layout()
    
    # Save and show
    plt.savefig('parameter_evolution.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    data_file = "data.txt"
    df = parse_data(data_file)
    visualize_data(df, 0)
    
    # Create plots if data was loaded successfully
    # if df is not None:
    #     create_plots(df)
    #     print("Plots saved as PNG files in current directory")