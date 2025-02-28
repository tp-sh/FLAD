import numpy as np
import sklearn.datasets as data
import hdbscan

moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
test_data = np.vstack([moons, blobs])

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True).fit(test_data)

print(clusterer.labels_)
