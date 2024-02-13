import matplotlib.pyplot as plt
import mlflow
import numpy as np
import openjij as oj
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

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

n = 2  # 任意の次元数に変更可能
combined_data = []

# 各領域について、隣接するn個のbinを組み合わせたベクトルを作成
for i in range(len(binarized_data[regions[0]]) - n + 1):
    combined_bin = []
    for j in range(n):
        combined_bin.extend([binarized_data[region][i + j] for region in regions])
    combined_data.append(np.array(combined_bin))

combined_data = np.array(combined_data)


size = 18  # 18次元のベクトルを扱う


class QuantumBoltzmannMachine:
    def __init__(self, size, sampler_choice="SQA"):
        self.size = size
        self.W = np.zeros((size, size))
        self.b = np.zeros(size)
        if sampler_choice == "SQA":
            self.sampler = oj.SQASampler()
        elif sampler_choice == "SA":
            self.sampler = oj.SASampler()
        else:
            raise ValueError("Unsupported sampler choice.")

    def sample_dwave(self, num_reads):
        linear, quadratic = self.to_ising()
        bqm = oj.BinaryQuadraticModel(linear, quadratic, oj.SPIN)
        sampleset = self.sampler.sample(bqm, num_reads=num_reads)
        return sampleset.record.sample

    def to_ising(self):
        linear = {i: -self.b[i] for i in range(self.size)}
        quadratic = {
            (i, j): -self.W[i, j]
            for i in range(self.size)
            for j in range(i + 1, self.size)
        }
        return linear, quadratic

    def fit(self, data, shot, epochs, learning_rate, num_samples):
        energy_history = []
        for epoch in tqdm(range(epochs), desc=f"Training Progress for Shot {shot + 1}"):
            data_corr = np.mean([np.outer(d, d) for d in data], axis=0)
            sampleset = self.sample_dwave(num_samples)
            model_samples = sampleset
            model_corr = np.mean([np.outer(m, m) for m in model_samples], axis=0)
            self.W += learning_rate * (data_corr - model_corr)
            np.fill_diagonal(self.W, 0)
            self.b += learning_rate * (
                np.mean(data, axis=0) - np.mean(model_samples, axis=0)
            )
            energy = np.mean(
                [
                    -0.5 * np.dot(d.T, np.dot(self.W, d)) - np.dot(self.b.T, d)
                    for d in data
                ]
            )
            energy_history.append(energy)
        return energy_history


# サンプリングされた解の分布の可視化
def plot_sample_distribution(samples, title):
    samples_df = pd.DataFrame(samples)
    sample_counts = samples_df.apply(pd.Series.value_counts, normalize=True).fillna(0)
    sample_counts.plot(kind="bar", stacked=True)
    plt.title(title)
    plt.xlabel("Node Index")
    plt.ylabel("Frequency")
    plt.legend(["-1", "1"], title="State")


def plot_weight_distribution(weights, title):
    plt.hist(weights.flatten(), bins=50)
    plt.title(title)
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")


# 相関行列の可視化関数
def plot_correlation_matrix(matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="coolwarm")
    plt.title(title)


def calculate_correlation_matrix(W, block_size=size // n):
    num_blocks = W.shape[0] // block_size
    correlation_matrix = np.zeros((num_blocks, num_blocks))

    for i in range(num_blocks):
        for j in range(num_blocks):
            block_i = W[
                i * block_size : (i + 1) * block_size,
                i * block_size : (i + 1) * block_size,
            ]
            block_j = W[
                j * block_size : (j + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            correlation, _ = pearsonr(block_i.flatten(), block_j.flatten())
            correlation_matrix[i, j] = correlation

    return correlation_matrix


def perform_clustering(correlation_matrix):
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        affinity="precomputed",
        linkage="complete",
    )
    # 相関行列を距離行列に変換
    distance_matrix = 1 - np.abs(correlation_matrix)
    clustering_model.fit(distance_matrix)
    return clustering_model.labels_


# MLflow実験の開始
with mlflow.start_run(run_name="72Dimension Boltzmann Machine Training"):
    num_shots = 1
    epochs = 100
    learning_rate = 0.009
    num_samples = 1000
    num_reads = 100

    sampler_choice = "SQA"  # "SQA" または "SA" を選択

    # QuantumBoltzmannMachine インスタンスの初期化
    qbm = QuantumBoltzmannMachine(size=size, sampler_choice=sampler_choice)

    # パラメータの記録
    mlflow.log_param("num_shots", num_shots)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_samples", num_samples)
    mlflow.log_param("num_reads", num_reads)
    mlflow.log_param("size", size)
    mlflow.log_param("sampler_choice", sampler_choice)

    for shot in range(num_shots):
        print(f"Starting training for 72Dim shot {shot + 1}")
        energy_history = qbm.fit(
            combined_data,
            shot=shot,
            epochs=epochs,
            learning_rate=learning_rate,
            num_samples=num_samples,
        )

        # エネルギー履歴のグラフ
        plt.figure(figsize=(10, 6))
        plt.plot(energy_history)
        plt.xlabel("Epoch")
        plt.ylabel("Energy")
        plt.title(f"Energy History - 72Dim Shot {shot + 1}")
        mlflow.log_figure(plt.gcf(), f"energy_history_72Dim_shot_{shot + 1}.png")
        plt.close()

        # 重みの可視化
        plt.figure(figsize=(10, 10))
        sns.heatmap(qbm.W, cmap="coolwarm", square=True)
        # plt.colorbar()
        plt.title("Weights Heatmap")
        mlflow.log_figure(plt.gcf(), f"weights_72Dim_shot_{shot + 1}.png")
        plt.close()

        # バイアスの可視化
        plt.figure(figsize=(10, 6))
        plt.bar(range(qbm.size), qbm.b)
        plt.title("Biases")
        mlflow.log_figure(plt.gcf(), f"biases_72Dim_shot_{shot + 1}.png")
        plt.close()

        # 時空間相関の分析
        correlation_matrix = calculate_correlation_matrix(qbm.W, n)
        labels = perform_clustering(correlation_matrix)
        plot_correlation_matrix(
            correlation_matrix, "Correlation Matrix - Shot " + str(shot + 1)
        )
        mlflow.log_figure(
            plt.gcf(), "correlation_matrix_shot_" + str(shot + 1) + ".png"
        )
        plt.close()
