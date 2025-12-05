import numpy as np

def greedy_solve(dist_matrix, time_windows, service_times, priorities):
    """
    Greedy Nearest Neighbor approach with Time Windows.
    
    Args:
        dist_matrix (np.array): Distance matrix between nodes.
        time_windows (list of tuples): (start, end) for each node.
        service_times (list): Service time for each node.
        priorities (list): Priority of each node (unused in basic distance greedy, but available).
        
    Returns:
        list: Ordered list of node indices (including depot at start/end).
        float: Total distance.
        float: Total time.
    """
    num_nodes = len(dist_matrix)
    unvisited = set(range(1, num_nodes)) # 0 is depot
    current_node = 0
    current_time = 0
    route = [0]
    total_dist = 0
    
    # Depot time window usually starts at 0
    
    while unvisited:
        best_node = -1
        min_dist = float('inf')
        
        # Find nearest feasible neighbor
        for node in unvisited:
            dist = dist_matrix[current_node][node]
            arrival_time = current_time + dist # Assuming speed=1 for simplicity: dist=time
            start_window, end_window = time_windows[node]
            
            # Check feasibility
            if arrival_time <= end_window:
                # We can arrive before it closes. 
                # If we arrive early, we wait.
                wait_time = max(0, start_window - arrival_time)
                # Heuristic cost: Distance (Standard Greedy). 
                # Could add wait_time to cost to avoid long waits.
                cost = dist 
                
                if cost < min_dist:
                    min_dist = cost
                    best_node = node
        
        if best_node != -1:
            # Move to best node
            dist = dist_matrix[current_node][best_node]
            arrival_time = current_time + dist
            start_window, end_window = time_windows[best_node]
            wait_time = max(0, start_window - arrival_time)
            
            current_time = arrival_time + wait_time + service_times[best_node]
            total_dist += dist
            current_node = best_node
            route.append(best_node)
            unvisited.remove(best_node)
        else:
            # No feasible node found (or dead end due to time windows)
            # For this simple greedy, we just stop or return to depot
            break
            
    # Return to depot
    return_dist = dist_matrix[current_node][0]
    total_dist += return_dist
    route.append(0)
    
    return route, total_dist, current_time + return_dist

if __name__ == "__main__":
    # Test logic
    pass
