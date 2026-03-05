print('--- RAPIDS Smoke Test ---')
try:
    import cudf
    import cuml
    import numpy as np
    from cuml.cluster import KMeans
    data = np.random.rand(100, 5).astype('float32')
    gdf = cudf.DataFrame(data)
    print(f'✓ cuDF: Successfully created GPU DataFrame (Shape: {gdf.shape})')
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(gdf)
    print('✓ cuML: Successfully ran KMeans on GPU')
except Exception as e:
    print(f'✗ Test Failed: {e}')
