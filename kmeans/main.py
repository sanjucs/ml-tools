import torch
import matplotlib.pyplot as plt

from kmeans import KMEANS

NUM_SAMPLES_PER_GAUSSIAN = 1000
NUM_GAUSSIAN = 3
NUM_CLUSTERS = 5

mean = torch.tensor([
  [-5, 0],
  [7, -10],
  [9, 11]
])

covariance_diag = torch.tensor([
  [2, 11],
  [3, 2],
  [3, 3]
])

def generate_samples(mean, variance):
  x = torch.randn(NUM_SAMPLES_PER_GAUSSIAN, 2)
  x[:, 0] = x[:, 0] * variance[0] + mean[0]
  x[:, 1] = x[:, 1] * variance[1] + mean[1]
  return x

g0 = generate_samples(mean[0], covariance_diag[0])
g1 = generate_samples(mean[1], covariance_diag[1])
g2 = generate_samples(mean[2], covariance_diag[2])

X = torch.cat((g0, g1, g2))
X = X[torch.randperm(len(X))]

kmeans = KMEANS(11)
cluster_X = kmeans.fit(X)

plt.subplot(1, 2, 1)
for i in range(NUM_GAUSSIAN):
  plt.scatter(g0[: , 0], g0[:, 1], color='C0')
  plt.scatter(g1[: , 0], g1[:, 1], color='C1')
  plt.scatter(g2[: , 0], g2[:, 1], color='C2')
plt.title('Input samples')

plt.subplot(1, 2, 2)
for idx, cluster in enumerate(kmeans.clusters):
  cluster = torch.stack(cluster)
  plt.scatter(cluster[: , 0], cluster[:, 1])
  plt.scatter(kmeans.centroid[idx][0], kmeans.centroid[idx][1], c='black')
plt.title('K means after {0:d} steps'.format(kmeans.step))

plt.savefig('output.png')
