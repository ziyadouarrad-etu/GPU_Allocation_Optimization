from src.optimizer import solve_gpu_allocation
from src.visualizer import generate_reports
import json
import pandas as pd
import os

def load_data():
    # Get the directory where main.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    # Load JSON Configuration
    config_path = os.path.join(data_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load CSVs
    costs = pd.read_csv(os.path.join(data_dir, 'costs.csv')).values.tolist()
    avail = pd.read_csv(os.path.join(data_dir, 'availability.csv'))['hours'].tolist()
    demands = pd.read_csv(os.path.join(data_dir, 'demands.csv'))['units'].tolist()
    
    T = config['T_factor']
    times = [[T/6, T/3.6, T], [T/6, T/3.6, T], [T/7, T/4, T]]
    
    return config, costs, times, avail, demands

def main():
    config, costs, times, avail, demands = load_data()
    status, allocation, total_cost = solve_gpu_allocation(costs, times, avail, demands)
    
    if status == 'Optimal':
        print(f"Optimal Solution Found! Total Cost: ${total_cost:,.2f}")
        generate_reports(allocation, costs, times, avail, config['models'], config['gpus'])
    else:
        print("Optimization failed to find a valid solution.")

if __name__ == "__main__":
    main()