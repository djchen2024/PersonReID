import faiss
def diverse_subset_sort_kmeans(data, m, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Sorts a vector dataset into diverse subsets of size m using k-means clustering.
    Args:
        data: A PyTorch tensor or NumPy array representing the vector dataset (n x d).
        m: The desired size of each diverse subset.
    Returns:
        A PyTorch tensor or NumPy array representing the sorted dataset. Returns the same 
        type as the input data. Returns None if input is invalid.
    """
    t0 = time.time()
    if not isinstance(data, (torch.Tensor, np.ndarray)):
        print("Error: Input data must be a PyTorch tensor or NumPy array.")
        return None
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy().astype('float32')
        data_is_tensor = True
    else:
        data_np = data.astype('float32')
        data_is_tensor = False
    n, d = data_np.shape
    if m > n:
        print("Error: Subset size m cannot be larger than the total number of vectors n.")
        return None
    if m <= 0:
        print("Error: Subset size m must be a positive integer.")
        return None
    sorted_indices = []
    remaining_indices = list(range(n))
    num_subsets = (n + m - 1) // m
    # for _ in tqdm(range(num_subsets)):
    for _ in range(num_subsets):    
        subset_indices = []
        if len(remaining_indices) <= m:
            subset_indices = remaining_indices[:]
            remaining_indices = []
        else:
            remaining_data = data_np[remaining_indices]
            # FAISS KMeans - Correct and Simplified
            n_clusters = m
            if device == "cuda":
                res = faiss.StandardGpuResources()
                kmeans = faiss.Kmeans(d, n_clusters, min_points_per_centroid=1, verbose=False)
            else:
                kmeans = faiss.Kmeans(d, n_clusters, min_points_per_centroid=1, verbose=False)
            kmeans.train(remaining_data)
            _, I = kmeans.index.search(remaining_data, 1)  # Search for 1 nearest centroid for each point
            closest_indices_within_remaining = []
            for i in range(n_clusters):
                cluster_indices = np.where(I == i)[0]
                if len(cluster_indices) > 0:
                    closest_indices_within_remaining.append(cluster_indices[np.argmin(kmeans.index.search(remaining_data[cluster_indices],1)[0])])
            closest_indices_within_remaining = np.array(closest_indices_within_remaining).reshape(-1,1)
            for idx in closest_indices_within_remaining:
                subset_indices.append(remaining_indices[idx[0]])
            remaining_indices = [i for i in remaining_indices if i not in subset_indices]
        sorted_indices.extend(subset_indices)
    t1 = time.time()
    print('Spends {:.2f} seconds for diverse sorting.'.format(t1 - t0))
    return sorted_indices