import matplotlib.pyplot as plt
import networkx as nx

def plot_routes(df, routes, title="Route Visualization"):
    """
    Plots the routes on a 2D plane.
    routes: list of lists (each sublist is a route/cluster).
    """
    plt.figure(figsize=(10, 8))
    
    # Plot all nodes
    plt.scatter(df['x'], df['y'], c='gray', alpha=0.5, label='Nodes')
    
    # Plot Depot
    depot = df[df['is_depot'] == True]
    plt.scatter(depot['x'], depot['y'], c='red', s=100, marker='D', label='Depot')
    
    colors = plt.cm.get_cmap('tab10', len(routes))
    
    for i, route in enumerate(routes):
        # route is list of indices
        route_points = df.loc[route]
        plt.plot(route_points['x'], route_points['y'], marker='o', linestyle='-', linewidth=2, color=colors(i), label=f'Route {i+1}')
        
        # Add arrows or order?
        
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Save
    import os
    if not os.path.exists("output"):
        os.makedirs("output")
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(f"output/{filename}")
    print(f"Saved plot to output/{filename}")

def plot_comparison(baseline_metrics, ml_metrics):
    """
    Compares Distance/Time.
    metrics = {'distance': X, 'time': Y}
    """
    labels = ['Distance', 'Time']
    baseline_vals = [baseline_metrics['distance'], baseline_metrics['time']]
    ml_vals = [ml_metrics['distance'], ml_metrics['time']]
    
    x = range(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], baseline_vals, width, label='Greedy Baseline')
    ax.bar([i + width/2 for i in x], ml_vals, width, label='ML Optimized')
    
    ax.set_ylabel('Value')
    ax.set_title('Optimization Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    import os
    if not os.path.exists("output"):
        os.makedirs("output")
    plt.savefig("output/comparison_chart.png")
    print("Saved comparison chart to output/comparison_chart.png")
