# メトロポリス-ヘイスティング・アルゴリズムを使用して事後分布からのサンプリングをシミュレートする
# 事後分布は、正規分布の平均と分散の事前分布を使用して、正規分布の平均と分散を推定する
# 事前分布は、平均が0で分散が100の正規分布を使用する

# In[ ]:   # 事後分布のパラメータを推定するためのデータを生成
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

# 事前分布のパラメータ
mu_0 = 0
sigma_0 = 100

# 事後分布のパラメータを推定するためのデータを生成
N = 100
mu = 10
sigma = 5
data = np.random.normal(mu, sigma, N)

# 事後分布のパラメータを推定するためのデータを描画
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20)
plt.xlabel("x")
plt.ylabel("Frequency")
plt.title("事後分布のパラメータを推定するためのデータ")
plt.show()

# 事後分布のパラメータを推定するためのデータの平均と分散を計算
mu_hat = np.mean(data)
sigma_hat = np.var(data)
print(f"事後分布のパラメータを推定するためのデータの平均: {mu_hat:.2f}")
print(f"事後分布のパラメータを推定するためのデータの分散: {sigma_hat:.2f}")

# 事後分布のパラメータを推定するためのデータの平均と分散を描画
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20)
plt.axvline(mu_hat, color="red", linestyle="dashed", linewidth=2)
plt.axvline(mu_hat + sigma_hat, color="red", linestyle="dashed", linewidth=2)
plt.axvline(mu_hat - sigma_hat, color="red", linestyle="dashed", linewidth=2)
plt.xlabel("x")
plt.ylabel("Frequency")
plt.title("事後分布のパラメータを推定するためのデータ")
plt.show()
