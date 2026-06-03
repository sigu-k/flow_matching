# Claude Code 引き継ぎ:CIFAR-10 学習時p(t) × 推論配置の実装

## リポジトリ
/home/jovyan/work/srv21/flow_matching/examples/image/
(base 環境で動作。torchdiffeq, torchmetrics[image] インストール済み)

## 目的
flow matching で「学習時のtimestep分布」と「推論時のステップ配置」を
5パターンずつ切り替え可能にし、5×5の総当たりでFIDを比較する。
研究の問い:中央集中など特定のtを重視する学習・推論に優位性があるか。

## 統一する density 関数(最重要・検算済み)

全配置を rho(t) = 1 + A * shape(t)、A=1.0 で統一する。
学習も推論も同じ rho(t) を共有すること(これが本実験の肝)。

| 配置名 | shape(t)        | rho(t)       |
|--------|-----------------|--------------|
| uniform| 0               | 1            |
| center | sin(pi*t)       | 1+sin(pi*t)  |
| both   | 1 - sin(pi*t)   | 2-sin(pi*t)  |
| data   | 1 - t           | 2-t          |
| noise  | t               | 1+t          |

実装方法(逆変換サンプリング):
1. grid = linspace(0,1,10000)
2. rho = shape_fn(grid, dist, A=1.0)
3. cdf = cumsum(rho); cdf /= cdf[-1]
4. 学習側:u = rand(n) を searchsorted(cdf, u) で grid に逆引き → t
5. 推論側:targets = linspace(0,1,num_steps) を同様に逆引き → t
           ただし推論時のみ両端を t[0]=0.0, t[-1]=1.0 に固定

共通関数として shape_fn と build_cdf を1箇所に実装し、
学習・推論の両方から呼ぶこと(重複実装しない)。

## 検算済みの期待される形(実装後、必ずこれと一致するか確認)

学習側サンプリング(n=200k, 10区間, 各%):
- uniform: [9 9 9 9 10 10 10 9 10 10]
- center : [7 8 10 11 11 12 11 10 8 7]
- both   : [13 11 9 8 7 7 8 9 11 13]
- data   : [12 12 11 11 10 9 9 8 7 7]
- noise  : [7 7 8 8 9 10 10 11 12 12]

推論側配置(num_steps=50, 10区間のステップ数, 両端固定):
- uniform: [5 5 5 5 5 5 5 5 5 5]
- center : [4 4 5 6 6 6 6 5 4 4]
- both   : [7 6 4 4 4 4 4 4 6 7]
- data   : [7 6 6 5 5 5 4 4 4 4]
- noise  : [4 4 4 4 5 5 5 6 6 7]

全配置で全区間>0(学習可能性)を満たすこと。

## 実装箇所

### 学習側:training/train_loop.py
- 75行目あたり `t = torch.torch.rand(samples.shape[0]).to(device)` を
  density方式に置き換え
- 既存の skewed_timestep_sample(26-33行)と --skewed_timesteps 分岐(92-95行)は
  使わない。新しい --timestep_dist 引数(choices: uniform/center/both/data/noise)で切替
- t の生成は build_cdf + searchsorted で行う

### 推論側:training/eval_loop.py
- 151行目あたりの args.edm_schedule 分岐の近くで、ステップ配置を生成している箇所を特定
- num_steps 個のステップ時刻を、density方式の配置(両端固定)に置き換え
- 新しい --sampling_dist 引数(choices: uniform/center/both/data/noise)で切替
- ODE ソルバーに渡す時刻系列をこの配置にする

### 引数:train_arg_parser.py
- --timestep_dist (default=uniform, choices=[uniform,center,both,data,noise])
- --sampling_dist (default=uniform, choices=[uniform,center,both,data,noise])

## 重要な制約
- 作業前に git commit でセーブポイントを作る(srv21 のリポジトリ)
- 各ファイル編集前に diff を提示し承認を得てから書き込む
- A=1.0 は固定でよい(将来パラメータ化したいが今は定数でOK)
- uniform 配置は既存の linspace と完全一致すること(後方互換の確認)
- 実装後、上記「検算済みの期待される形」と一致するか検証スクリプトで確認すること

## 動作確認
実装後、まず1パターンで --test_run:
python train.py --dataset=cifar10 --batch_size=64 --class_drop_prob=1.0 \
  --cfg_scale=0.0 --timestep_dist=center --test_run

## 本実験の進め方(参考、実装後)
- 学習5回:--timestep_dist を5パターン、各 --epochs=100
- 各学習済みモデルに対し推論5パターン:--sampling_dist を5パターン
- 計25通りの FID を測定(--compute_fid --ode_method heun2 --ode_options '{"nfe":50}' --use_ema)
- 学習は約2.6時間/パターン(RTX 4090)