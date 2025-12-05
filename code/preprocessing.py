import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

def preprocess_data(df):
    """
    Preprocesses the data for ML models and optimization.
    
    Args:
        df (pd.DataFrame): Input dataframe with nodes.
        
    Returns:
        pd.DataFrame: Scaled dataframe (with normalized Coordinates).
        np.array: Distance matrix (original scale).
    """
    # 1. Normalize coordinates for ML (clustering)
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[['x_scaled', 'y_scaled']] = scaler.fit_transform(df[['x', 'y']])
    
    # 2. Calculate Distance Matrix (Euclidean on original coordinates)
    # We use original coordinates for the actual cost calculation
    coords = df[['x', 'y']].values
    distance_matrix = cdist(coords, coords, metric='euclidean')
    
    return df_scaled, distance_matrix

if __name__ == "__main__":
    try:
        df = pd.read_csv("../dataset/synthetic_data.csv")
        df_scaled, dist_matrix = preprocess_data(df)
        print("Distance Matrix Shape:", dist_matrix.shape)
        print(df_scaled[['x', 'y', 'x_scaled', 'y_scaled']].head())
    except FileNotFoundError:
        print("Run data_generator.py first.")
