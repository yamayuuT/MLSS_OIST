import matplotlib.pyplot as plt
import mlflow
import numpy as np
import openjij as oj
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
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


size = 9 * n  # 18次元のベクトルを扱う


class QuantumBoltzmannMachine:
    def __init__(self, size):
        self.size = size
        # 小さなランダム値で初期化
        # self.Wをランダムな値で初期化し、対称行列にする
        self.W = np.random.uniform(low=-0.001, high=0.001, size=(size, size))
        self.W = (self.W + self.W.T) / 2  # 対称性を保証
        np.fill_diagonal(self.W, 0)  # 対角要素を0に設定
        self.b = np.random.uniform(low=-0.001, high=0.001, size=size)
        # np.fill_diagonal(self.W, 0)  # 対角要素を0に設定
        self.sampler = oj.SASampler()

    def energy(self, x):
        return -np.dot(x.T, np.dot(self.W, x)) - np.dot(self.b.T, x)

    def generate_custom_schedule(self):
        # カスタムのアニーリングスケジュールを定義
        # 例: [(0.0, beta_start), (t1, beta_mid), (t2, beta_mid), (t_final, beta_end)]
        # ここで、betaは逆温度を指します
        schedule = [(0.0, 1.0), (1.0, 0.6), (9.0, 0.6), (10.0, 1.0)]
        plot_annealing_schedule(schedule)
        return schedule

    def sample_ising(self, num_samples):
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
        """
        if schedule is not None:
            beta_schedule = [
                (t, 1.0 / T) for t, T in schedule
            ]  # OpenJijでのbetaスケジュール
            sampleset = self.sampler.sample_ising(
                linear, quadratic, num_reads=num_reads, schedule=beta_schedule
            )
        else:
            sampleset = self.sampler.sample_ising(
                linear, quadratic, num_reads=num_reads
            )
        """

        # SASamplerでサンプリング
        sampleset = self.sampler.sample_ising(linear, quadratic, num_reads=num_samples)
        samples = sampleset.record.sample  # サンプリング結果

        # サンプルを{-1, 1}の形式に変換
        samples = 2 * samples - 1
        return samples

    def fit(
        self, data, epochs, learning_rate, num_samples, n_lowest_samples=1500, desc=None
    ):
        energy_history = []
        for epoch in tqdm(
            range(epochs), desc="Training Quantum Boltzmann Machine", unit="epoch"
        ):
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
            np.fill_diagonal(self.W, 0)
            self.b += learning_rate * (
                np.mean(data, axis=0) - np.mean(lowest_samples, axis=0)
            )
            energy = np.mean([self.energy(d) for d in data])
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
with mlflow.start_run(run_name="72Dimension Boltzmann Machine Training Optimized"):
    epochs = 200
    num_samples = 10000

    # ベイズ最適化のための目的関数
    def objective_function(learning_rate):
        # ベイズ最適化の各ステップで評価する学習率を出力
        print(f"\nEvaluating learning rate: {learning_rate[0]}")
        qbm = QuantumBoltzmannMachine(size=size)
        # トレーニングプロセスのカスタムメッセージ
        energy_history = qbm.fit(
            combined_data,
            epochs=epochs,
            learning_rate=learning_rate[0],
            num_samples=num_samples,
            desc=f"Evaluating learning rate: {learning_rate[0]}",  # tqdmの説明を動的に更新
        )
        return energy_history[-1]

    # 学習率のパラメータ空間を定義
    space = [Real(1e-10, 1e-1, name="learning_rate")]

    # ベイズ最適化の実行
    @use_named_args(space)
    def objective(**params):
        return objective_function([params["learning_rate"]])

    # ベイズ最適化の前後にメッセージを出力
    print("Bayesian Optimization Process: Starting\n")
    res = gp_minimize(objective, space, n_calls=40, random_state=0)
    print("\nBayesian Optimization Process: Completed")
    print(f"Optimal learning rate: {res.x[0]}")
    print(f"Optimal energy: {res.fun}")

    # ベイズ最適化で見つけた最適な学習率を使用
    optimal_learning_rate = res.x[0]  # これはベイズ最適化セクションから得られる値です

    num_shots = 5
    learning_rate = optimal_learning_rate  # 最適化された学習率を使用
    num_reads = 100

    # QuantumBoltzmannMachine インスタンスの初期化
    qbm = QuantumBoltzmannMachine(size=size)

    # パラメータの記録
    mlflow.log_param("num_shots", num_shots)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_samples", num_samples)
    mlflow.log_param("num_reads", num_reads)
    mlflow.log_param("size", size)

    for shot in range(num_shots):
        print(
            f"\nStarting training for BO+SASampler shot {shot + 1} with optimized learning rate: {optimal_learning_rate}"
        )
        energy_history = qbm.fit(
            combined_data,
            epochs=epochs,
            learning_rate=learning_rate,
            num_samples=num_samples,
            desc=f"Training QBM - Shot {shot + 1}",  # tqdmの説明を更新して明確にする
        )
        print(f"Training Process: Completed for Shot {shot + 1}")

        # エネルギー履歴のグラフ
        plt.figure(figsize=(10, 6))
        plt.plot(energy_history)
        plt.xlabel("Epoch")
        plt.ylabel("Energy")
        plt.title(
            f"Energy History - 72Dim Shot {shot + 1} with Optimized Learning Rate"
        )
        mlflow.log_figure(
            plt.gcf(), f"energy_history_optimized_72Dim_shot_{shot + 1}.png"
        )
        plt.close()

        # 重みの可視化
        plt.figure(figsize=(10, 10))
        sns.heatmap(qbm.W, cmap="coolwarm", square=True)
        plt.title("Weights Heatmap - Optimized Learning Rate")
        mlflow.log_figure(plt.gcf(), f"weights_optimized_72Dim_shot_{shot + 1}.png")
        plt.close()

        # バイアスの可視化
        plt.figure(figsize=(10, 6))
        plt.bar(range(qbm.size), qbm.b)
        plt.title("Biases - Optimized Learning Rate")
        mlflow.log_figure(plt.gcf(), f"biases_optimized_72Dim_shot_{shot + 1}.png")
        plt.close()

        # 時空間相関の分析
        correlation_matrix = calculate_correlation_matrix(qbm.W, n)
        labels = perform_clustering(correlation_matrix)
        plot_correlation_matrix(
            correlation_matrix,
            f"Correlation Matrix - Shot {shot + 1} with Optimized Learning Rate",
        )
        mlflow.log_figure(
            plt.gcf(), f"correlation_matrix_optimized_shot_{shot + 1}.png"
        )
        plt.close()
