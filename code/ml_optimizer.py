import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from greedy_solver import greedy_solve

def ml_optimize(df, dist_matrix, n_clusters=3, random_state=42):
    """
    Cluster-first, Route-second approach.
    1. Cluster nodes using K-Means on coordinates.
    2. Solve TSP/VRP for each cluster using Greedy.
    3. Combine results.
    """
    # Filter out depot for clustering
    customers = df[df['is_depot'] == False].copy()
    depot = df[df['is_depot'] == True].iloc[0]
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    # Using spatial coordinates for clustering
    # If priorities are important, we could weight samples or include priority as feature?
    # For now, spatial clustering is standard for routing.
    customers['cluster'] = kmeans.fit_predict(customers[['x', 'y']])
    
    total_dist = 0
    total_time = 0
    full_route = []
    
    # Solve for each cluster
    for cid in range(n_clusters):
        cluster_nodes = customers[customers['cluster'] == cid]
        
        # We need to map global indices to local matrix indices or extract sub-matrix
        # Easiest: Create a sub-problem
        # Node IDs in the dataframe are 1-based usually, but matrix is 0-based index.
        # We need to map dataframe index to matrix index.
        # Assuming dataframe is sorted by index? Data generator creates normal index 0 to N.
        # Let's rely on dataframe index matching matrix index if we include depot.
        
        # Sub-problem indices: [0 (Depot)] + [Cluster Nodes indices]
        node_indices = [0] + cluster_nodes.index.tolist()
        
        # Sub-distance matrix
        # ix_ allows selecting rows/cols by label. But `dist_matrix` is numpy.
        # We need integer positions. 
        sub_matrix = dist_matrix[np.ix_(node_indices, node_indices)]
        
        # Also need sub-time_windows, sub-service_times
        sub_time_windows = df.loc[node_indices, ['time_window_start', 'time_window_end']].values
        sub_service_times = df.loc[node_indices, 'service_time'].values
        sub_priorities = df.loc[node_indices, 'priority'].values
        
        # Solve
        route_local_idx, r_dist, r_time = greedy_solve(sub_matrix, sub_time_windows, sub_service_times, sub_priorities)
        
        # Map local indices back to global
        route_global = [node_indices[i] for i in route_local_idx]
        
        total_dist += r_dist
        total_time += r_time # This assumes parallel execution if multiple vehicles, or sum if sequential.
        # If sequential single vehicle, time would accumulate. 
        # But clustering implies dividing work (Fleet). 
        # If single vehicle, we'd need to link clusters. 
        # Let's assume we sum distances (cost) and are evaluating efficient routing.
        
        full_route.append(route_global)
        
    return full_route, total_dist, total_time

if __name__ == "__main__":
    pass
