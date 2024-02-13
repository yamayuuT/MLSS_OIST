import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 与えられたデータ
orders = np.array([0, 1, 2, 3, 4, 5, 6, 7])
times = np.array([90, 210, 810, 1845, 3720, 10920, 32760, 98460])

# 指数関数の定義
def exp_func(x, a, b):
    return a * (b ** x)

# パラメータの最適化
params, _ = curve_fit(exp_func, orders, times)

# 推定されたパラメータでオーダー6の計算時間を推定
predicted_time = exp_func(7, *params)

print(f"推定されたオーダー6の計算時間: {predicted_time:.2f} ms")

# グラフ描画
plt.figure(figsize=(10, 6))
plt.plot(orders, times, 'o', label='実際のデータ')
plt.plot(np.arange(8), exp_func(np.arange(8), *params), '-', label='フィットされた曲線')
plt.xlabel('指数オーダー')
plt.ylabel('計算時間 (ms)')
plt.title('指数オーダーと計算時間の関係')
plt.legend()
plt.show()
