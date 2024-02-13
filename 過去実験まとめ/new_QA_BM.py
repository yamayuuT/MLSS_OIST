import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import KFold
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
from tqdm import tqdm
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from minorminer import find_embedding

# ExcelファイルからEEGデータの読み込み
original_eeg_data = pd.read_excel("/Users/yamayuu/BoltzmannMachine_pre/EEG_all.xlsx")

# 初期設定
bin_size = 10
regions = ["F3", "F4", "Fz", "C3", "C4", "P3", "P4", "A1", "A2"]
sampling_rate = 250

# データの前処理
binarized_data = {}
for region in regions:
    c_data = original_eeg_data[region].to_numpy()
    smoothed_c_data = np.convolve(c_data, np.ones(bin_size) / bin_size, mode="same")
    binarized_c_data = np.where(c_data > smoothed_c_data, 1, -1)
    binarized_data[region] = binarized_c_data

# 隣接するbinを組み合わせて72次元ベクトルを作成
combined_data = []
for i in range(len(binarized_data[regions[0]]) - 7):
    combined_bin = np.array([binarized_data[region][i + j] for region in regions for j in range(8)])
    combined_data.append(combined_bin)
combined_data = np.array(combined_data)

class QuantumBoltzmannMachine:
    def __init__(self, size, token):
        self.size = size
        self.W = np.zeros((size, size))
        self.b = np.zeros(size)
        self.token = token

    def to_ising(self):
        linear = {i: -self.b[i] for i in range(self.size)}
        quadratic = {(i, j): -self.W[i, j] for i in range(self.size) for j in range(i+1, self.size)}
        return linear, quadratic

    def sample_dwave(self, num_reads=100):
        linear, quadratic = self.to_ising()
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, dimod.SPIN)
        dw_sampler = DWaveSampler(token=self.token)
        sampler = EmbeddingComposite(dw_sampler)
        sampleset = sampler.sample(bqm, num_reads=num_reads)
        return sampleset

    def fit(self, data, shot, epochs, learning_rate, num_samples, lambda_l2, lambda_l1):
        energy_history = []
        for epoch in tqdm(range(epochs), desc=f"Training Progress for Shot {shot + 1}"):
            data_corr = np.mean([np.outer(d, d) for d in data], axis=0)
            sampleset = self.sample_dwave(num_samples)
            model_samples = np.array([list(sample.values()) for sample in sampleset.samples()])
            model_corr = np.mean([np.outer(m, m) for m in model_samples], axis=0)
            self.W += learning_rate * (data_corr - model_corr)
            np.fill_diagonal(self.W, 0)
            self.b += learning_rate * (np.mean(data, axis=0) - np.mean(model_samples, axis=0))
            energy = np.mean([-np.dot(d.T, np.dot(self.W, d)) - np.dot(self.b.T, d) for d in data])
            energy_history.append(energy)
        return energy_history

# インスタンスの作成とトレーニングの実行
token = "DEV-263b4b057d6b25ccfa4ec7421800fec1846b3423"
qbm = QuantumBoltzmannMachine(size=72, token=token)

num_shots = 2
epochs = 200
learning_rate = 0.05
num_samples = 100
lambda_l2 = 0.001
lambda_l1 = 0.001

for shot in range(num_shots):
    print(f"Starting training for 72Dim shot {shot + 1}")
    energy_history = qbm.fit(combined_data, shot=shot, epochs=epochs, learning_rate=learning_rate, num_samples=num_samples, lambda_l2=lambda_l2, lambda_l1=lambda_l1)
    # エネルギー履歴のグラフを保存
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history)
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.title(f"Energy History - 72Dim Shot {shot + 1}")
    plt.savefig(f"energy_history_72Dim_shot_{shot + 1}.png")
    plt.close()

# 重みとバイアスの可視化
plt.figure(figsize=(10, 6))
plt.imshow(qbm.W, cmap="coolwarm")
plt.colorbar()
plt.title("Weights")
plt.savefig("weights_72Dim.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(range(qbm.size), qbm.b)
plt.title("Biases")
plt.savefig("biases_72Dim.png")
plt.close()
