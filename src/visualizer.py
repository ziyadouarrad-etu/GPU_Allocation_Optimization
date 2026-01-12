import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def generate_reports(allocation, costs, time_matrix, availability, models, gpus):
    """
    Generates and saves all project visualizations using absolute paths.
    """
    # 1. Establish Absolute Paths
    # Finds the directory of the current file (src/) and moves up one level
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base_dir, 'Plots')
    
    # Ensure the directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    solution = np.array(allocation)
    costs_arr = np.array(costs)
    time_arr = np.array(time_matrix)

    # Helper function to save plots with absolute paths
    def save_plot(filename):
        save_path = os.path.join(plots_dir, filename)
        plt.savefig(save_path)
        plt.close()

    # --- 1. Bar Plot: Model Deployment on GPUs ---
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    index = np.arange(len(gpus))
    for i in range(len(models)):
        plt.bar(index + i*bar_width, solution[i], bar_width, label=models[i])
    plt.xlabel('GPUs')
    plt.ylabel('Number of Models Deployed')
    plt.title('Model Deployment Distribution across GPUs')
    plt.xticks(index + bar_width, gpus)
    plt.legend()
    plt.tight_layout()
    save_plot('model_deployment_distribution.png')

    # --- 2. Pie Chart: Cost Distribution by Model ---
    cost_by_model = [sum(solution[i][j] * costs[i][j] for j in range(len(gpus))) for i in range(len(models))]
    plt.figure(figsize=(8, 8))
    plt.pie(cost_by_model, labels=models, autopct='%1.1f%%', startangle=140)
    plt.title('Total Operational Cost Distribution by Model Type')
    save_plot('cost_distribution_by_model.png')

    # --- 3. Heatmap: Production Allocation Matrix ---
    plt.figure(figsize=(10, 7))
    heatmap_data = pd.DataFrame(solution, index=models, columns=gpus)
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Number of Models'})
    plt.title('Production Allocation Matrix (Heatmap)')
    plt.xlabel('GPU Type')
    plt.ylabel('Model Type')
    save_plot('allocation_heatmap.png')

    # --- 4. Bar Chart: GPU Time Utilization ---
    time_used = np.sum(solution * time_arr, axis=0)
    utilization = (time_used / np.array(availability)) * 100
    plt.figure(figsize=(10, 6))
    plt.bar(gpus, utilization, color='skyblue')
    plt.axhline(100, color='red', linestyle='--', label='Max Capacity')
    plt.xlabel('GPU Type')
    plt.ylabel('Utilization (%)')
    plt.title('Hardware Resource Utilization (Time)')
    plt.ylim(0, 110)
    plt.legend()
    save_plot('gpu_time_utilization.png')

    # --- 5. Average Cost per Model Type ---
    total_demands = np.sum(solution, axis=1)
    avg_cost_per_model = [cost_by_model[i] / total_demands[i] if total_demands[i] > 0 else 0 for i in range(len(models))]
    plt.figure(figsize=(10, 6))
    plt.bar(models, avg_cost_per_model, color='lightgreen')
    plt.xlabel('Model Type')
    plt.ylabel('Average Cost ($)')
    plt.title('Economic Efficiency: Average Cost per Model Type')
    save_plot('average_cost_per_model.png')

    print(f"✅ Success! Graphs saved in: {plots_dir}")