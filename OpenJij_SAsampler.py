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

n = 16  # 任意の次元数に変更可能
combined_data = []

# 各領域について、隣接するn個のbinを組み合わせたベクトルを作成
for i in range(len(binarized_data[regions[0]]) - n + 1):
    combined_bin = []
    for j in range(n):
        combined_bin.extend([binarized_data[region][i + j] for region in regions])
    combined_data.append(np.array(combined_bin))

combined_data = np.array(combined_data)


size = 9 * n  # 18次元のベクトルを扱う


class QuantumBoltzmannMachine:
    def __init__(self, size, schedule):
        self.size = size
        # ウェイトをランダムに初期化し、対称行列にする
        W = np.random.randn(size, size) / np.sqrt(size)  # He初期化の原理に基づく
        self.W = (W + W.T) / 2  # Wが対称行列になるように
        # バイアスをゼロで初期化（バイアスの初期化に関しては、しばしばゼロからスタートすることが一般的です）
        # バイアスをランダムに初期化
        self.b = np.random.randn(size) / np.sqrt(size)
        # np.fill_diagonal(self.W, 0)  # 対角要素を0に設定
        self.schedule = schedule
        self.sampler = oj.SASampler()

    def energy(self, x):
        return -np.dot(x.T, np.dot(self.W, x)) - np.dot(self.b.T, x)

    def sample_ising(self, num_reads):
        # イジングモデルのパラメータを準備
        linear = {i: -self.b[i] for i in range(self.size)}
        quadratic = {
            (i, j): -self.W[i, j]
            for i in range(self.size)
            for j in range(i + 1, self.size)
        }
        # デバッグ情報の出力
        # print("Linear:", linear)
        # print("Quadratic:", quadratic)

        # SASamplerでサンプリング
        sampleset = self.sampler.sample_ising(
            linear,
            quadratic,
            num_reads=num_reads,
            schedule=self.schedule,
            seed=seed,
            num_sweeps=num_sweeps,
        )
        samples = sampleset.record.sample  # サンプリング結果

        # サンプルを{-1, 1}の形式に変換
        samples = 2 * samples - 1
        return samples

    def fit(self, data, epochs, learning_rate, num_samples, n_lowest_samples=1000):
        energy_history = []
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            data_corr = np.mean([np.outer(d, d) for d in data], axis=0)

            # サンプリング
            samples = self.sample_ising(num_samples)

            # 各サンプルのエネルギーを計算
            energies = np.array([self.energy(s) for s in samples])

            # エネルギーが最も低いn個のサンプルを選択
            lowest_samples_indices = np.argsort(energies)[:n_lowest_samples]
            lowest_samples = samples[lowest_samples_indices]

            # 選択したサンプルに基づいてモデル相関を計算
            model_corr = np.mean([np.outer(s, s) for s in lowest_samples], axis=0)

            # パラメータの更新
            self.W += learning_rate * (data_corr - model_corr)
            np.fill_diagonal(self.W, 0)  # 対角成分を0に設定
            self.b += learning_rate * (
                np.mean(data, axis=0) - np.mean(lowest_samples, axis=0)
            )

            # エネルギーの計算を修正
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


def plot_annealing_schedule(schedule):
    times = [point[0] for point in schedule]
    values = [point[1] for point in schedule]

    plt.figure(figsize=(10, 6))
    plt.plot(times, values, marker="o")
    plt.title("Reverse Annealing Schedule")
    plt.xlabel("Annealing Time (microseconds)")
    plt.ylabel("Execution Degree")
    plt.ylim(0, 1.2)  # y軸の範囲を0から1.0に設定
    plt.grid(True)
    mlflow.log_figure(plt.gcf(), "annealing_schedul_history_NDim_shot.png")
    # plt.show()


# MLflow実験の開始
with mlflow.start_run(run_name="SASampler Boltzmann Machine Training"):
    num_shots = 1
    epochs = 200
    learning_rate = 0.0004
    num_samples = 2000
    num_reads = 1
    num_sweeps = 1
    seed = 0

    # sampler_choice = "SQA"  # "SQA" または "SA" を選択

    # カスタマイズされたアニーリングスケジュールを定義
    custom_schedule = [[10, 1], [8, 5], [3, 15], [0.5, 20]]

    # QuantumBoltzmannMachine インスタンスの初期化時にスケジュールを渡す
    qbm = QuantumBoltzmannMachine(size=size, schedule=custom_schedule)

    # パラメータの記録
    mlflow.log_param("num_shots", num_shots)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_samples", num_samples)
    mlflow.log_param("num_reads", num_reads)
    mlflow.log_param("size", size)
    mlflow.log_param("seed", seed)
    mlflow.log_param("num_sweeps", num_sweeps)
    # mlflow.log_param("sampler_choice", sampler_choice)

    for shot in range(num_shots):
        print(f"Starting training for SASampler shot {shot + 1}")
        energy_history = qbm.fit(
            combined_data,
            epochs=epochs,
            learning_rate=learning_rate,
            num_samples=num_samples,
        )

        # エネルギー履歴のグラフ
        plt.figure(figsize=(10, 6))
        plt.plot(energy_history)
        plt.xlabel("Epoch")
        plt.ylabel("Energy")
        plt.title(f"Energy History - SASampler Shot {shot + 1}")
        mlflow.log_figure(plt.gcf(), f"energy_history_SASampler_shot_{shot + 1}.png")
        plt.close()

        # 重みの可視化
        plt.figure(figsize=(10, 10))
        sns.heatmap(qbm.W, cmap="coolwarm", square=True)
        # plt.colorbar()
        plt.title("Weights Heatmap")
        mlflow.log_figure(plt.gcf(), f"weights_SASampler_shot_{shot + 1}.png")
        plt.close()

        # バイアスの可視化
        plt.figure(figsize=(10, 6))
        plt.bar(range(qbm.size), qbm.b)
        plt.title("Biases")
        mlflow.log_figure(plt.gcf(), f"biases_SASampler_shot_{shot + 1}.png")
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
