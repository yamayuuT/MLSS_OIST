import numpy as np
import pandas as pd

import streamlit as st

# 以下、ExtendedBoltzmannMachineクラスとその他の関数定義
# ...（クラスと関数の定義をここに挿入）...

# from itertools
# ExcelファイルからEEGデータの読み込み
# original_eeg_data = pd.read_excel("/Users/yamayuu/BoltzmannMachine_pre/EEG_all.xlsx")


def energy_function(x, W, b):
    return -np.dot(x.T, np.dot(W, x)) - np.dot(b.T, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Streamlitアプリのメイン関数
def main():
    st.title("Extended Boltzmann Machine Experiment")

    # ファイルアップロード
    uploaded_file = st.file_uploader("Upload EEG Data", type=["xlsx"])
    if uploaded_file is not None:
        original_eeg_data = pd.read_excel(uploaded_file)

    # 初期設定
    bin_size = 10
    regions = ["F3", "F4", "Fz", "C3", "C4", "P3", "P4", "A1", "A2"]
    # sampling_rate = 250

    # データの前処理
    binarized_data = {}
    for region in regions:
        c_data = original_eeg_data[region].to_numpy()
        smoothed_c_data = np.convolve(c_data, np.ones(bin_size) / bin_size, mode="same")
        binarized_c_data = np.where(c_data > smoothed_c_data, 1, -1)
        binarized_data[region] = binarized_c_data

    n = 14  # 任意の次元数に変更可能
    combined_data = []

    # 各領域について、隣接するn個のbinを組み合わせたベクトルを作成
    for i in range(len(binarized_data[regions[0]]) - n + 1):
        combined_bin = []
        for j in range(n):
            combined_bin.extend([binarized_data[region][i + j] for region in regions])
        combined_data.append(np.array(combined_bin))

    combined_data = np.array(combined_data)

    size = 126  # 18次元のベクトルを扱う

    # 実験設定
    num_shots = st.sidebar.number_input("Number of Shots", 1, 10, 1)
    epochs = st.sidebar.number_input("Number of Epochs", 10, 100, 50)
    learning_rate = st.sidebar.number_input("Learning Rate", 0.01, 1.0, 0.05, 0.01)
    num_samples = st.sidebar.number_input("Number of Samples", 10, 1000, 20)
    lambda_l2 = st.sidebar.number_input("Lambda L2", 0.0, 1.0, 0.001, 0.001)
    lambda_l1 = st.sidebar.number_input("Lambda L1", 0.0, 1.0, 0.001, 0.001)
    size = n * 9
    n = st.sidebar.number_input("Number of Shots", 1, 20, 1)

    if st.button("Run Experiment") and uploaded_file is not None:
        run_experiment(
            data,
            num_shots,
            epochs,
            learning_rate,
            num_samples,
            lambda_l2,
            lambda_l1,
            size,
        )


def run_experiment(
    num_shots, epochs, learning_rate, num_samples, lambda_l2, lambda_l1, size
):
    # ここに実験コードを挿入

    # 結果の表示
    st.success("Experiment completed!")


# Streamlitのウィジェットを用いてconfigを更新


# Streamlitアプリを実行
if __name__ == "__main__":
    main()
