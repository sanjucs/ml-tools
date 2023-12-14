import matplotlib.pyplot as plt
import torch

class Kmeans:
  def __init__(self, k, criterion='L2', threshold=1e-3):
    self.k = k
    self.criterion = criterion
    self.threshold = threshold
    self.is_training = True
    self.step = 0

  def plot(self):
    plt.clf()
    for idx, cluster in enumerate(self.clusters):
      cluster = torch.stack(cluster)
      plt.scatter(cluster[: , 0], cluster[:, 1])
      plt.scatter(self.centroid[idx][0], self.centroid[idx][1], c='black')
    plt.title('K means after {0:d} steps'.format(self.step))
    plt.pause(1)

  def update_centroid(self):
    distance = 0
    for i in range(len(self.clusters)):
      new_mean = torch.mean(torch.stack(self.clusters[i]), dim=0)

      if self.criterion == 'L2':
        distance += torch.norm(self.centroid[i] - new_mean)
      else:
        raise NotImplementedError

      self.centroid[i] = new_mean
    distance = distance / len(self.centroid)

    if distance < self.threshold:
      self.is_training = False

  def init_clusters(self):
    self.clusters = [[] for _ in range(self.k)]

  def get_cluster_distances(self, x):
    return torch.norm(self.centroid - x, dim=1)

  def update(self, data):
    self.init_clusters()
    for x in data:
      cluster_idx = torch.argmin(self.get_cluster_distances(x))
      self.clusters[cluster_idx].append(x)
    self.step += 1
    self.update_centroid()
    self.plot()
  
  def fit(self, X):
    assert self.k < len(X), 'k more than number of unique data points'
    unique_indices = torch.randperm(len(X))[:self.k]
    self.centroid = X[unique_indices]

    while self.is_training:
      self.update(X)

    self.plot()

if __name__ == '__main__':
  kmeans = Kmeans(10)
  X = torch.randn(10, 2)
  kmeans.fit(X)
