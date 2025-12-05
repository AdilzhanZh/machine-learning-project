# Schedule & Route Optimization

This project implements a Route Optimization system combining a Greedy Algorithm with Machine Learning (Clustering).

## Project Structure
- `code/`: Contains all source code.
  - `data_generator.py`: Generates synthetic customer data with Time Windows.
  - `preprocessing.py`: Normalizes data for ML.
  - `greedy_solver.py`: Implements a Greedy Nearest Neighbor approach with time constraints.
  - `ml_optimizer.py`: Implements K-Means Clustering + Greedy Solving per cluster.
  - `visualizer.py`: Generates route plots.
  - `main.py`: Main entry point to run the simulation.
- `dataset/`: Stores generated CSV data.
- `output/`: Stores result visualizations.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python code/main.py
   ```

## Results
The system compares a single-route Baseline Greedy approach against a Cluster-First (ML) approach.
- **Baseline**: Minimizes distance for a single route but may fail to cover all nodes if constraints are tight.
- **ML Optimization**: Splits the problem into clusters (simulating multiple vehicles/routes), ensuring better coverage and organized routing, though potentially higher total distance due to multiple trips.

Results (PNG plots) are saved in the `output/` directory.
