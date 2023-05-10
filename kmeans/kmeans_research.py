import torch

class KMEANS:
  def __init__(self, k, threshold=1e-3):
    self.k = k
    self.threshold = threshold
    self.centroid = torch.randn(k, 2)

  def init_clusters(self):
    self.clusters = [[] for _ in range(self.k)]

  def update_mean(self):
    loss = 0
    for i in range(len(self.clusters)):
      if i != []:
        new_mean = torch.mean(torch.stack(self.clusters[i]), dim=0)
        loss += torch.norm(self.centroid[i] - new_mean)
        self.centroid[i] = new_mean
    loss = loss / len(self.centroid)
    print(loss)
    if loss < self.threshold:
      print("Stop")
      import matplotlib.pyplot as plt

      plt.scatter(torch.stack(self.clusters[0])[: , 0], torch.stack(self.clusters[0])[:, 1], color='C1')
      plt.scatter(torch.stack(self.clusters[1])[: , 0], torch.stack(self.clusters[1])[:, 1], color='C2')

      plt.title('Input samples')
      plt.show()
      # plt.savefig('samples.png')
      exit()
  def l2_norm(self, x):
    return torch.norm(self.centroid - x, dim=0)

  def update(self, data):
    self.init_clusters()
    for x in data:
      c = torch.argmin(self.l2_norm(x))
      self.clusters[c].append(x)
    self.update_mean()
  
  def fit(self, X):
    value = True
    while value:
      self.update(X)

kmeans = KMEANS(2)
data = torch.randn(10000, 2)
kmeans.fit(data)

