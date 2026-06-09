import matplotlib.pyplot as plt
import torch

# 提示されたモジュールから関数をインポート（同じディレクトリにある想定）
from timestep_density import DIST_CHOICES, sample_timesteps


def visualize_timestep_densities(num_samples=100000, bins=100, A=1.0):
    """各サンプリング戦略のタイムステップ密度を横並びのヒストグラムで可視化する。"""
    num_dists = len(DIST_CHOICES)

    # 論文のFigure 3のレイアウト（横並び）を模してフィギュアを作成
    fig, axes = plt.subplots(
        1, num_dists, figsize=(4 * num_dists, 3.5), sharey=True
    )

    # DIST_CHOICES の各分布についてループ処理
    for idx, dist in enumerate(DIST_CHOICES):
        ax = axes[idx]

        # 1. 提示されたコードの関数を用いて、指定した分布からn個のタイムステップをサンプリング
        with torch.no_grad():
            t_samples = sample_timesteps(num_samples, dist, A=A)

        # PyTorchテンソルをNumPy配列に変換
        t_samples_np = t_samples.cpu().numpy()

        # 2. ヒストグラムの描画
        # density=True にすることで、縦軸をカウントではなく確率密度（論文のDensity表記）にする
        ax.hist(
            t_samples_np,
            bins=bins,
            range=(0.0, 1.0),
            density=True,
            color="skyblue",
            edgecolor="none",
            alpha=0.8,
        )

        # 3. グラフの装飾（論文のスタイルに準拠）
        ax.set_title(f"{dist.capitalize()} (A={A})", fontsize=12)
        ax.set_xlabel("Time (t)", fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.6)

        # 一番左のグラフにのみ縦軸のラベルを表示
        if idx == 0:
            ax.set_ylabel("Density", fontsize=11)

    plt.tight_layout()

    # 画像として保存、および画面表示
    output_filename = "timestep_density_comparison.png"
    plt.savefig(output_filename, dpi=300)
    print(f"プロットを '{output_filename}' として保存した。")
    plt.show()


if __name__ == "__main__":
    # サンプリング数を多め（10万枚）に設定して綺麗な密度曲線を描画
    visualize_timestep_densities(num_samples=100000, bins=100, A=1.0)