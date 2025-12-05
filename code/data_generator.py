import numpy as np
import pandas as pd

def generate_synthetic_data(num_nodes=50, x_range=(0, 100), y_range=(0, 100), depot_pos=(50, 50), random_seed=42):
    """
    Generates synthetic data for route optimization.
    
    Args:
        num_nodes (int): Number of customer nodes (excluding depot).
        x_range (tuple): Range of x coordinates.
        y_range (tuple): Range of y coordinates.
        depot_pos (tuple): (x, y) coordinates of the depot.
        random_seed (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: DataFrame containing node information.
    """
    np.random.seed(random_seed)
    
    # Generate coordinates
    x_coords = np.random.uniform(x_range[0], x_range[1], num_nodes)
    y_coords = np.random.uniform(y_range[0], y_range[1], num_nodes)
    
    # Generate time windows (start time between 0-480 min (8 hours), duration 30-120 min)
    # Global time horizon: 0 to 600
    avg_service_time = 10
    
    # Earliest start times
    start_times = np.random.uniform(0, 480, num_nodes)
    # Latest end times (must be after start + service)
    end_times = start_times + np.random.uniform(30, 120, num_nodes)
    
    # Priorities (1-5, higher is more important)
    priorities = np.random.randint(1, 6, num_nodes)
    
    data = {
        'node_id': np.arange(1, num_nodes + 1),
        'x': x_coords,
        'y': y_coords,
        'service_time': [avg_service_time] * num_nodes,
        'time_window_start': start_times,
        'time_window_end': end_times,
        'priority': priorities,
        'is_depot': False
    }
    
    df = pd.DataFrame(data)
    
    # Add depot
    depot_row = pd.DataFrame([{
        'node_id': 0,
        'x': depot_pos[0],
        'y': depot_pos[1],
        'service_time': 0,
        'time_window_start': 0,
        'time_window_end': 1000, # Large window for depot
        'priority': 0,
        'is_depot': True
    }])
    
    df = pd.concat([depot_row, df], ignore_index=True)
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(df.head())
    df.to_csv("../dataset/synthetic_data.csv", index=False)
    print("Data saved to dataset/synthetic_data.csv")
