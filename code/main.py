import pandas as pd
import numpy as np
from data_generator import generate_synthetic_data
from preprocessing import preprocess_data
from greedy_solver import greedy_solve
from ml_optimizer import ml_optimize
from visualizer import plot_routes, plot_comparison
import os

def main():
    # 1. Data Generation
    print("Generating Data...")
    if not os.path.exists("../dataset"):
        os.makedirs("../dataset")
        
    df = generate_synthetic_data(num_nodes=50)
    df.to_csv("../dataset/synthetic_data.csv", index=False)
    
    # 2. Preprocessing
    print("Preprocessing...")
    df_scaled, dist_matrix = preprocess_data(df)
    
    # 3. Baseline Greedy
    print("Running Baseline Greedy...")
    # Baseline treats everything as one big problem
    # Extract arrays
    time_windows = df[['time_window_start', 'time_window_end']].values
    service_times = df['service_time'].values
    priorities = df['priority'].values
    
    base_route, base_dist, base_time = greedy_solve(dist_matrix, time_windows, service_times, priorities)
    print(f"Baseline - Dist: {base_dist:.2f}, Time: {base_time:.2f}")
    
    plot_routes(df, [base_route], title="Baseline Greedy Route")
    
    # 4. ML Optimized (Clustering)
    print("Running ML Optimization...")
    ml_routes, ml_dist, ml_time = ml_optimize(df, dist_matrix, n_clusters=3)
    print(f"ML Optimized - Dist: {ml_dist:.2f}, Time: {ml_time:.2f}")
    
    plot_routes(df, ml_routes, title="ML Cluster Optimized Routes")
    
    # 5. Comparison
    plot_comparison(
        {'distance': base_dist, 'time': base_time},
        {'distance': ml_dist, 'time': ml_time}
    )
    
    print("Done! Check .png files for results.")

if __name__ == "__main__":
    main()
