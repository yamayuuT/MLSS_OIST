import matplotlib.pyplot as plt
import mlflow
import networkx as nx
import numpy as np

# import cupy as cp
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
from sklearn.model_selection import KFold
from tqdm import tqdm

# from itertools
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

"""
# 隣接するbinを組み合わせて72次元ベクトルを作成
combined_data = []
for i in range(len(binarized_data[regions[0]]) - 5):
    combined_bin = np.array([binarized_data[region][i + j] for region in regions for j in range(6)])
    combined_data.append(combined_bin)
combined_data = np.array(combined_data)
"""
"""#以下ミスった時用の基準コード
combined_data = []
for i in range(len(binarized_data[regions[0]]) - 1):
    combined_bin = np.array(
        [binarized_data[region][i] for region in regions]
        + [binarized_data[region][i + 1] for region in regions]
    )
    combined_data.append(combined_bin)

combined_data = np.array(combined_data)
"""

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


def energy_function(x, W, b):
    return -np.dot(x.T, np.dot(W, x)) - np.dot(b.T, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ExtendedBoltzmannMachine:
    def __init__(self, size):
        self.size = size
        # ウェイトをランダムに初期化し、対称行列にする
        W = np.random.randn(size, size) / np.sqrt(size)  # He初期化の原理に基づく
        self.W = (W + W.T) / 2  # Wが対称行列になるように
        # バイアスをゼロで初期化（バイアスの初期化に関しては、しばしばゼロからスタートすることが一般的です）
        self.b = np.zeros(size)

    def energy(self, x):
        return -np.dot(x.T, np.dot(self.W, x)) - np.dot(self.b.T, x)

    def mcmc_sample(self, num_samples):
        samples = np.zeros((num_samples, self.size))
        x = np.random.choice([1, -1], size=self.size)
        for i in range(num_samples):
            for j in range(self.size):
                energy_diff = 2 * (np.dot(self.W[j, :], x) + self.b[j]) * x[j]
                if energy_diff < 0 or np.random.rand() < np.exp(-energy_diff):
                    x[j] *= -1
            samples[i, :] = x
        return samples

    def fit(self, data, shot, epochs, learning_rate, num_samples, lambda_l2, lambda_l1):
        energy_history = []
        weight_history = []  # 重み行列の履歴を保持するリスト

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            data_corr = np.mean([np.outer(d, d) for d in data], axis=0)
            model_samples = self.mcmc_sample(num_samples)
            model_corr = np.mean([np.outer(m, m) for m in model_samples], axis=0)

            self.W += learning_rate * (data_corr - model_corr)
            np.fill_diagonal(self.W, 0)
            self.b += learning_rate * (
                np.mean(data, axis=0) - np.mean(model_samples, axis=0)
            )

            # L2正則化項（Ridge回帰）
            if lambda_l2 > 0.0:
                self.W -= learning_rate * lambda_l2 * self.W

            # L1正則化項（LASSO回帰）
            if lambda_l1 > 0.0:
                self.W -= learning_rate * lambda_l1 * np.sign(self.W)

            # 重み行列を記録
            weight_history.append(self.W.copy())
            # エポックごとのエネルギー値を計算して記録
            energy = np.mean([self.energy(d) for d in data])
            energy_history.append(energy)
            """
            if epoch % 10 == 0:  # 例えば10エポックごとに出力
                print(f"Epoch {epoch}: Weights:\n{self.W}")
                print(f"Epoch {epoch}: Biases:\n{self.b}") 
            """

        # エネルギー履歴のグラフを保存してMLflowに記録
        energy_history_filename = f"energy_history_18Dim_shot_{shot + 1}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(energy_history)
        plt.xlabel("Epoch")
        plt.ylabel("Energy")
        plt.title(f"Energy History - 18Dim Shot {shot + 1}")
        # mlflowに直接グラフを保存
        mlflow.log_figure(plt.gcf(), energy_history_filename)
        plt.close()
        # 最初と最後の重み行列をヒートマップとして表示
        self.plot_weights_heatmap(
            weight_history[0], f"Initial Weights - Shot {shot + 1}"
        )
        self.plot_weights_heatmap(
            weight_history[-1], f"Final Weights - Shot {shot + 1}"
        )

    # 学習メソッドにクロスバリデーションを追加
    def cross_validate(
        self, data, k=5, lambda_l2_range=[0.01, 0.1], lambda_l1_range=[0.01, 0.1]
    ):
        kf = KFold(n_splits=k)
        best_score = float("inf")
        best_lambda_l2 = None
        best_lambda_l1 = None

        for lambda_l2 in lambda_l2_range:
            for lambda_l1 in lambda_l1_range:
                scores = []
                for train_index, test_index in kf.split(data):
                    self.W = np.zeros((self.size, self.size))  # 重みをリセット
                    self.b = np.zeros(self.size)  # バイアスをリセット
                    train_data, test_data = data[train_index], data[test_index]
                    self.fit(
                        train_data,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        num_samples=num_samples,
                        lambda_l2=lambda_l2,
                        lambda_l1=lambda_l1,
                    )
                    score = self.evaluate(test_data)  # モデルの評価メトリックを計算
                    scores.append(score)

                average_score = np.mean(scores)
                if average_score < best_score:
                    best_score = average_score
                    best_lambda_l2 = lambda_l2
                    best_lambda_l1 = lambda_l1

        return best_lambda_l2, best_lambda_l1, best_score

    def evaluate(self, data):
        """
        モデルの評価を行います。評価指標としては、データとモデルによって生成されたサンプル間の
        カルバック・ライブラー発散 (KL発散) を使用します。これにより、モデルがデータをどれだけ
        よく表現しているかを測定します。
        """
        # モデルによるサンプルの生成
        model_samples = self.mcmc_sample(len(data))

        # データとモデルサンプルの確率分布を推定
        data_distribution, _ = np.histogram(
            data, bins=np.arange(-1.5, 2.0, 1.0), density=True
        )
        model_distribution, _ = np.histogram(
            model_samples, bins=np.arange(-1.5, 2.0, 1.0), density=True
        )

        # 0確率を避けるための微小量を追加
        data_distribution += 1e-10
        model_distribution += 1e-10

        # KL発散の計算
        kl_divergence = entropy(data_distribution, model_distribution)
        return kl_divergence

    def reconstruct(self, data):
        """
        データの再構成を行います。各可視ノードの状態を、隠れノードの状態を条件としてサンプリングします。
        ここでは、モデルの重みとバイアスを使って、元のデータと同じサイズの再構成データを生成する
        プロセスを実装します。
        """
        reconstructed_data = np.zeros_like(data)
        for i, sample in enumerate(data):
            # 隠れノードの状態をサンプリング
            hidden_states = np.sign(np.dot(self.W.T, sample) + self.b)
            # 可視ノードの再構成
            reconstructed_data[i] = np.sign(np.dot(self.W, hidden_states) + self.b)
        return reconstructed_data

    """
    def perform_clustering(self):
        # リンケージ行列を生成
        Z = linkage(self.W, "ward")

        perform_clustering_filename = f"perform_clustering_18Dim_shot_{shot + 1}.png"

        # デンドログラムを描画
        plt.figure(figsize=(10, 8))
        dendrogram(Z)
        plt.title("Dendrogram for the Clustering of Nodes")
        plt.xlabel("Node Index")
        plt.ylabel("Distance")
        # plt.show()
        mlflow.log_figure(plt.gcf(), perform_clustering_filename)
        plt.close()

        # クラスタ数を指定して階層的クラスタリングを実行
        n_clusters = 2  # 例としてクラスタ数を2に設定
        clustering_model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="euclidean", linkage="ward"
        )
        clustering_model.fit(self.W)

        # クラスタラベルを取得
        cluster_labels = clustering_model.labels_
        return cluster_labels

        # def perform_network_analysis(self):
        # 隣接行列からNetworkXグラフを生成
        G = nx.from_numpy_matrix(self.W)

        # グラフの基本的な特性を計算
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        average_degree = float(2 * num_edges) / num_nodes
        density = nx.density(G)

        # 中心性指標を計算
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)

        # グラフの特性を記録
        network_properties = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "average_degree": average_degree,
            "density": density,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "closeness_centrality": closeness_centrality,
        }

        return network_properties
        """

    def sparsify_network(self, threshold=0.1):
        # 重み行列からスパースなグラフを作成
        G_sparse = nx.Graph()
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if abs(self.W[i, j]) > threshold:
                    G_sparse.add_edge(i, j, weight=self.W[i, j])

        return G_sparse

    def analyze_network_properties(self, G):
        # スパース化されたグラフの特性を計算
        properties = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "average_clustering": nx.average_clustering(G, weight="weight"),
            "average_shortest_path_length": nx.average_shortest_path_length(G),
        }

        # スモールワールド性の評価（オプション）
        try:
            properties["small_world_coefficient"] = nx.sigma(G, niter=100, nrand=10)
        except:
            properties["small_world_coefficient"] = None

        return properties

    def display_weights_and_biases(self, shot):
        weights_filename = f"weights_18Dim_shot_{shot + 1}.png"
        biases_filename = f"biases_18Dim_shot_{shot + 1}.png"

        max_abs_weight = np.max(np.abs(self.W))

        # 重みのグラフを保存
        plt.figure(figsize=(10, 6))
        plt.imshow(self.W, cmap="coolwarm", clim=(-max_abs_weight, max_abs_weight))
        plt.colorbar()
        plt.title(f"Weights - 18Dim Shot {shot + 1}")
        # plt.savefig(weights_filename)
        # plt.close()
        mlflow.log_figure(plt.gcf(), weights_filename)
        plt.close()

        # バイアスのグラフを保存
        plt.figure(figsize=(10, 6))
        plt.bar(range(self.size), self.b)
        plt.title(f"Biases - 18Dim Shot {shot + 1}")
        # plt.savefig(biases_filename)
        # plt.close()
        mlflow.log_figure(plt.gcf(), biases_filename)
        plt.close()

    def analyze_weights(self, shot):
        heatmap_filename = f"weights_heatmap_18Dim_shot_{shot + 1}.png"

        # 軸ラベルを準備
        axis_labels = regions * n  # 2時刻分のラベル

        # 重みの最大絶対値を基にカラースケールの範囲を設定
        max_abs_weight = np.max(np.abs(self.W))
        color_scale_range = (-max_abs_weight, max_abs_weight)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.W,
            annot=True,
            cmap="coolwarm",
            center=0,
            clim=color_scale_range,
            xticklabels=axis_labels,
            yticklabels=axis_labels,
        )
        plt.title(f"Weight Matrix Heatmap - 18Dim Shot {shot + 1}")
        plt.xlabel("i+1 time regions")
        plt.ylabel("i time regions")
        plt.tight_layout()  # ラベルが切れないように調整
        mlflow.log_figure(plt.gcf(), heatmap_filename)
        plt.close()

    def generate_samples(self, num_samples):
        return self.mcmc_sample(num_samples)

    def plot_weights_heatmap(self, weights, title):
        plt.figure(figsize=(8, 8))
        sns.heatmap(
            weights,
            cmap="coolwarm",
            square=True,
            clim=(-np.max(np.abs(self.W.copy())), np.max(np.abs(self.W.copy()))),
        )
        plt.title(title)
        mlflow.log_figure(plt.gcf(), f"{title.replace(' ', '_')}.png")
        plt.close()


# KLダイバージェンス計算関数
def calculate_kl_divergence(p, q):
    return entropy(p, q)


# JSダイバージェンス計算関数
def calculate_js_divergence(p, q):
    return jensenshannon(p, q)


# Wasserstein距離計算関数
def calculate_wasserstein_distance(p, q):
    return wasserstein_distance(p, q)


def plot_thresholded_heatmap(W, threshold=0.1):
    # 重み行列のヒートマップを生成
    plt.figure(figsize=(12, 10))
    sns.heatmap(W, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Weight Matrix Heatmap with Threshold {threshold}")
    plt.xlabel("i+1 time")
    plt.ylabel("i time")

    # スレッショルドを超える重みに基づいて線を描画
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if abs(W[i, j]) > threshold:
                plt.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color="blue")
                plt.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color="blue")

    plt.tight_layout()


num_shots = 1  # 例: 3回のショット
epochs = 50
learning_rate = 0.08
num_samples = 1000
# 学習の実行時に正則化の係数を渡す
lambda_l2 = 0.001  # L2正則化の係数
lambda_l1 = 0.001  # L1正則化の係数

# bm_size = 2**exponent

# processed_data = preprocess_data(original_eeg_data, regions, base_bin_size, exponent)

"""# サンプリング結果の解析
for sample in sqa_sampleset.samples():
    print("Sample:", sample)
    print("Energy:", sqa_sampleset.record.energy)
"""

mlflow.end_run()


# 親実行を開始
with mlflow.start_run(run_name="18Dimension Boltzmann Machine Training"):
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_samples", num_samples)
    mlflow.log_param("bin_size", bin_size)
    # mlflow.log_param("exponent", exponent)
    mlflow.log_param("lambda_l2", lambda_l2)
    mlflow.log_param("lambda_l1", lambda_l1)
    mlflow.log_param("size", size)

    for shot in range(num_shots):
        print(f"Starting training for 18Dim shot {shot + 1}")

        # 子実行を開始
        with mlflow.start_run(run_name=f"Shot {shot + 1}", nested=True):
            # bm = ExtendedBoltzmannMachine(size=size)
            # 既存のコードに追加
            bm = ExtendedBoltzmannMachine(size=size)
            bm.fit(
                combined_data,
                shot=shot,
                epochs=epochs,
                learning_rate=learning_rate,
                num_samples=num_samples,
                lambda_l2=lambda_l2,
                lambda_l1=lambda_l1,
            )
            # 学習とその他の処理を実行...
            # cluster_labels = bm.perform_clustering()
            # クラスタラベルをMLflowに記録
            # mlflow.log_param(f"cluster_labels_shot_{shot + 1}", cluster_labels.tolist())

            # ネットワーク解析の実行
            """
            network_properties = bm.perform_network_analysis()  # ネットワーク解析メソッドの呼び出し
            for key, value in network_properties.items():
                param_name = f"{key}_shot_{shot + 1}"  # 各ショットごとに異なるパラメータ名を使用
                # パラメータが既に存在するかどうかを確認
                existing_params = mlflow.get_run(child_run.info.run_id).data.params
                if param_name not in existing_params:
                    mlflow.log_param(param_name, value)
            """

            # G_sparse = bm.sparsify_network(threshold=0.1)
            # network_properties = bm.analyze_network_properties(G_sparse)
            # ネットワークの特性をMLflowに記録
            # for key, value in network_properties.items():
            # mlflow.log_param(f"{key}_shot_{shot + 1}", value)

            # パラメータの記録とグラフの表示・保存
            bm.display_weights_and_biases(shot)

            # 重み行列の分析
            bm.analyze_weights(shot)  # 引数 shot を渡す
            # サンプル生成と分布間距離の計算
            generated_samples = bm.generate_samples(1000)
            original_dist, _ = np.histogram(
                combined_data, bins=20, range=(-1, 1), density=True
            )
            generated_dist, _ = np.histogram(
                generated_samples, bins=20, range=(-1, 1), density=True
            )

            # 重み行列 W の例
            # W_example = bm.W  # 仮の重み行列

            # スレッショルドを0.1に設定してヒートマップを描画
            # plot_thresholded_heatmap(W_example, threshold=0.1)

            # mlflowにヒートマップを保存
            # heatmap_filename = "thresholded_heatmap.png"
            # mlflow.log_figure(plt.gcf(), heatmap_filename)
            # plt.close()

            # kl_divergence = calculate_kl_divergence(original_dist, generated_dist)
            # js_divergence = calculate_js_divergence(original_dist, generated_dist)
            """
            wasserstein_dist = calculate_wasserstein_distance(
                original_dist, generated_dist
            )

            print(f"KL Divergence (Shot {shot + 1}):", kl_divergence)
            print(f"JS Divergence (Shot {shot + 1}):", js_divergence)
            print(f"Wasserstein Distance (Shot {shot + 1}):", wasserstein_dist)
            mlflow.log_param("kl_divergence", kl_divergence)
            mlflow.log_param("js_divergence", js_divergence)
            mlflow.log_param("wasserstein_dist", wasserstein_dist)
            """

mlflow.end_run()
