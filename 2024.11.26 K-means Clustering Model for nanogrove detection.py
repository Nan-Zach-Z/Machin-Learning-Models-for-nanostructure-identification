import numpy as np
from scipy.ndimage import sobel
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_and_process_data_final(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = []
        for line in lines:
            if all(x.replace('.', '', 1).isdigit() or x == 'Bad' for x in line.strip().split()):
                row = []
                for value in line.strip().split():
                    if value == 'Bad':
                        row.append(np.nan)
                    else:
                        try:
                            row.append(float(value))
                        except ValueError:
                            row.append(np.nan)
                data.append(row)
        data = np.array(data, dtype=np.float64)
        mask = np.isnan(data)
        data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        sobel_x = sobel(data, axis=1)
        sobel_y = sobel(data, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return edge_magnitude
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Load and preprocess data
file_path_11 = '/mnt/data/11.ASC'
edge_data_11 = load_and_process_data_final(file_path_11)

if edge_data_11 is not None:
    # Applying k-means clustering
    flat_edge_data = edge_data_11.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(flat_edge_data)
    cluster_image = clusters.reshape(edge_data_11.shape)

    # Plotting the clustered image
    plt.figure(figsize=(10, 6))
    plt.imshow(cluster_image, cmap='viridis')
    plt.title('K-Means Clustering of Surface Features')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.colorbar(ticks=[0, 1])
    plt.show()

    # Groove parameter calculations using Otsu's method
    thresh = threshold_otsu(edge_data_11)
    binary_edges = edge_data_11 > thresh
    labeled_edges = label(binary_edges)
    properties = regionprops(labeled_edges, edge_data_11)
    groove_parameters = []
    for prop in properties:
        width = prop.bbox[3] - prop.bbox[1]  # max_col - min_col
        depth = prop.intensity_max - prop.intensity_min
        aspect_ratio = width / depth if depth != 0 else 0
        volume = prop.area * depth
        groove_parameters.append({
            'Width': width,
            'Depth': depth,
            'Aspect Ratio': aspect_ratio,
            'Volume': volume
        })
    for i, gp in enumerate(groove_parameters, 1):
        print(f"Groove {i}: Width = {gp['Width']} pixels, Depth = {gp['Depth']} units, Aspect Ratio = {gp['Aspect Ratio']:.2f}, Volume = {gp['Volume']} units^3")
else:
    print("Failed to load or process data for feature extraction.")
