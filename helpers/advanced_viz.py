import pandas as pd
import matplotlib.pyplot as plt

def plot_distance_histogram(pkl_file):
    df = pd.read_pickle(pkl_file)
    plt.hist(df[df['label']==0]['dist_to_normal_centroid'], bins=50, alpha=0.5, label='Normal')
    plt.hist(df[df['label']==1]['dist_to_normal_centroid'], bins=50, alpha=0.5, label='Anomaly')
    plt.title("Distance to Normal Centroid")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_distance_histogram('BGL_clustered_centroids.pkl')
    plot_distance_histogram('Thunderbird_clustered_centroids.pkl')