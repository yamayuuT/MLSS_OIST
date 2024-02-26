import matplotlib.pyplot as plt
import mlflow
import numpy as np

# import cupy as cp
import pandas as pd
from tqdm import tqdm

# from itertools

# ExcelファイルからEEGデータの読み込み
original_eeg_data = pd.read_excel(
    "/Users/yamayuu/Library/Mobile Documents/com~apple~CloudDocs/研究/BM_RNN/EEG_all.xlsx"
)


# 初期設定
bin_size = 10
regions = ["F3", "F4", "Fz", "C3", "C4", "P3", "P4", "A1", "A2"]
sampling_rate = 2

# データの前処理
binarized_data = {}

for region in regions:
    # 領域データの取り出し
    c_data = original_eeg_data[region].to_numpy()

    # 移動平均の計算
    smoothed_c_data = np.convolve(c_data, np.ones(bin_size) / bin_size, mode="same")

    # バイナライズ
    binarized_c_data = np.where(c_data > smoothed_c_data, 1, -1)

    binarized_data[region] = binarized_c_data


def energy_function(x, W, b):
    return -np.dot(x.T, np.dot(W, x)) - np.dot(b.T, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BoltzmannMachine:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))
        self.b = np.zeros(size)

    def energy(self, x):
        return -np.dot(x.T, np.dot(self.W, x)) - np.dot(self.b.T, x)

    def mcmc_sample(self, num_samples, initial_state=None):
        samples = np.zeros((num_samples, self.size))

        # 初期状態が与えられない場合は、固定された初期状態を使用
        if initial_state is None:
            initial_state = np.random.choice([1, -1], size=self.size)

        x = initial_state.copy()
        for i in range(num_samples):
            for j in range(self.size):
                x[j] = (
                    1
                    if np.random.rand() < self.sigmoid(np.dot(self.W[j], x) + self.b[j])
                    else -1
                )
            samples[i, :] = x
        return samples

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, data, epochs=100, learning_rate=0.1, num_samples=100, iterations=1):
        for iteration in range(iterations):
            energy_history = []
            bias_history = []

            for epoch in tqdm(range(epochs), desc=f"Training Progress - Iteration {iteration + 1}"):
                model_samples = self.mcmc_sample(num_samples)

                # 重みは更新しない
                # self.W += learning_rate * (data_corr - model_corr)
                # np.fill_diagonal(self.W, 0)

                # バイアスのみを更新
                self.b += learning_rate * (np.mean(data, axis=0) - np.mean(model_samples, axis=0))

                # エネルギーとバイアスの履歴を記録
                energy = np.mean([self.energy(d) for d in data])
                energy_history.append(energy)
                bias_history.append(self.b.copy())

            # エネルギー履歴のグラフを保存して記録
            plt.figure(figsize=(10, 6))
            plt.plot(energy_history)
            plt.xlabel("Epoch")
            plt.ylabel("Energy")
            plt.title("Learning Progress - Energy")
            plt.savefig(f"energy_history_{iteration + 1}.png")
            plt.close()
            mlflow.log_artifact(f"energy_history_{iteration + 1}.png")

            # バイアス履歴のグラフを保存して記録
            plt.figure(figsize=(10, 6))
            plt.imshow(np.array(bias_history).T, aspect='auto')
            plt.colorbar()
            plt.xlabel("Epoch")
            plt.ylabel("Bias Index")
            plt.title(f"Learning Progress - Biases")
            plt.savefig(f"biases_history_{iteration + 1}.png")
            plt.close()
            mlflow.log_artifact(f"biases_history_{iteration + 1}.png")


    def sample(self, num_samples):
        return self.mcmc_sample(num_samples)

    def display_weights_and_biases(self, shot, show_graphs=True):
        weights_filename = f"weights_shot_{shot + 1}.png"
        biases_filename = f"biases_shot_{shot + 1}.png"

        # 重みのグラフを保存
        plt.figure(figsize=(10, 6))
        plt.imshow(self.W, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.title(f"Weights - Shot {shot + 1}")
        plt.savefig(weights_filename)
        #if show_graphs:
            #plt.show()
        plt.close()
        mlflow.log_artifact(weights_filename)

        # バイアスのグラフを保存
        plt.figure(figsize=(10, 6))
        plt.bar(range(self.size), self.b)
        plt.title(f"Biases - Shot {shot + 1}")
        plt.savefig(biases_filename)
        #if show_graphs:
            #plt.show()
        plt.close()
        mlflow.log_artifact(biases_filename)


# 学習データの形式を整える
training_data = np.array(list(binarized_data.values())).T

# 複数回の独立した学習を行う
num_shots = 3  # 実行するショットの数
epochs = 200
learning_rate = 0.005
num_samples = 100

for shot in range(num_shots):
    print(f"Starting training for shot {shot + 1}")

    with mlflow.start_run(
        run_name=f"Boltzmann Machine Training - Shot {shot + 1}", nested=True
    ):
        bm = BoltzmannMachine(size=9)

        bm.fit(
            training_data,
            epochs=epochs,
            learning_rate=learning_rate,
            num_samples=num_samples,
        )

        # パラメータの記録
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("bin_size", bin_size)

        #mlflow.log_artifact("energy_history.png")
        #mlflow.log_artifact("weights.png")
        #mlflow.log_artifact("biases.png")

        # ショット数が多い場合はグラフを表示しない
        show_graphs = num_shots <= 3
        bm.display_weights_and_biases(shot, show_graphs=show_graphs)
