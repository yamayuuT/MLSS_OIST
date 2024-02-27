import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

# ExcelファイルからEEGデータの読み込み
original_eeg_data = pd.read_excel("/Users/yamayuu/BoltzmannMachine_pre/EEG_all.xlsx")

# 初期設定
bin_size = 10
regions = ["F3", "F4", "Fz", "C3", "C4", "P3", "P4", "A1", "A2"]

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


class ExtendedBoltzmannMachine:
    def __init__(self, size):
        self.size = size
        W = np.random.randn(size, size) / np.sqrt(size)
        self.W = (W + W.T) / 2
        self.b = np.random.randn(size) / np.sqrt(size)
        self.energy_history = []

    def energy(self, x):
        return -0.5 * np.dot(x.T, np.dot(self.W, x)) - np.dot(self.b.T, x)

    def mcmc_sample(self, num_samples):
        samples = np.zeros((num_samples, self.size))
        x = np.random.choice([1, -1], size=self.size)  # 初期状態

        for i in range(num_samples):
            for j in range(self.size):
                x_proposal = np.copy(x)
                x_proposal[j] = -x_proposal[j]  # 状態jの反転を提案
                energy_diff = self.energy(x_proposal) - self.energy(
                    x
                )  # エネルギー差ΔEの計算

                # 遷移確率 W(x|x') の計算
                # 反転がエネルギーを減少させる、または、確率 exp(-ΔE) に従って反転を受け入れる
                transition_prob = min(1, np.exp(-energy_diff))

                if np.random.rand() < transition_prob:
                    x = x_proposal  # 状態更新

            samples[i, :] = x

        return samples

    def fit(self, data, epochs, learning_rate, num_samples):
        self.energy_history = []
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            data_corr = np.mean([np.outer(d, d) for d in data], axis=0)
            model_samples = self.mcmc_sample(num_samples)
            model_corr = np.mean([np.outer(m, m) for m in model_samples], axis=0)

            self.W += learning_rate * (data_corr - model_corr)
            np.fill_diagonal(self.W, 0)  # 対角成分を0に設定して自己結合を防ぐ
            self.b += learning_rate * (
                np.mean(data, axis=0) - np.mean(model_samples, axis=0)
            )

            # エネルギーの計算を修正
            energy = np.mean(
                [
                    -0.5 * np.dot(d.T, np.dot(self.W, d)) - np.dot(self.b.T, d)
                    for d in data
                ]
            )
            self.energy_history.append(energy)

    def display_weights_and_biases(self, shot):
        weights_filename = f"weights_18Dim_shot_{shot + 1}.png"
        biases_filename = f"biases_18Dim_shot_{shot + 1}.png"

        # 重みのグラフを保存
        plt.figure(figsize=(10, 6))
        plt.imshow(
            self.W,
            cmap="coolwarm",
            clim=(-np.max(np.abs(self.W)), np.max(np.abs(self.W))),
        )
        plt.colorbar()
        plt.title(f"Weights - 18Dim Shot {shot + 1}")
        mlflow.log_figure(plt.gcf(), weights_filename)
        plt.close()

        # バイアスのグラフを保存
        plt.figure(figsize=(10, 6))
        plt.bar(range(self.size), self.b)
        plt.title(f"Biases - 18Dim Shot {shot + 1}")
        mlflow.log_figure(plt.gcf(), biases_filename)
        plt.close()


num_shots = 1
epochs = 200
learning_rate = 0.08
num_samples = 10000

with mlflow.start_run(run_name="MCMC Boltzmann Machine Training"):
    for shot in range(num_shots):
        print(f"Starting training for shot {shot + 1}")
        bm = ExtendedBoltzmannMachine(size=size)
        bm.fit(
            combined_data,
            epochs=epochs,
            learning_rate=learning_rate,
            num_samples=num_samples,
        )
        mlflow.log_param("num_shots", num_shots)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("size", size)

        # エネルギー履歴のグラフ
        plt.figure(figsize=(10, 6))
        plt.plot(bm.energy_history)
        plt.xlabel("Epoch")
        plt.ylabel("Energy")
        plt.title(f"Energy History - Shot {shot + 1}")
        mlflow.log_figure(plt.gcf(), f"energy_history_shot_{shot + 1}.png")
        plt.close()
        bm.display_weights_and_biases(shot)
