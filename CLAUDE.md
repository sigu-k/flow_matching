# flow_matching プロジェクト

Meta flow_matching を fork した CIFAR-10 用の研究コード。
timestep 分布(学習時 ρ(t) と推論時ステップ配置)を独立に振った 5×5 総当たり実験を進めている。

## 環境

- サーバ:srv21(RTX 4090 24GB)、base conda 環境
- リポジトリ:`/home/jovyan/work/srv21/flow_matching/`(editable install)
- PYTHONPATH:`/home/jovyan/work/srv21/flow_matching` を通す必要あり
- 追加パッケージ:torchdiffeq, torchmetrics[image], torch-fidelity
- 永続領域:`/home/jovyan/work/srv21/` 配下のみ(他はコンテナリセットで消える)

## 重要なファイル

- `training/timestep_density.py`:shape_fn / build_cdf / sample_timesteps / sampling_timesteps
- `training/train_loop.py`:学習側 t サンプリング
- `training/eval_loop.py`:推論側ステップ配置
- `train_arg_parser.py`:--timestep_dist / --sampling_dist(5 択)追加済み
- `output_<dist>/checkpoint.pth`:学習済みモデル(5 分布分、絶対に上書きしない)
- `fid_results/<train>__<sample>/`:推論結果(25 セル)

## 現在のタスク:5×5 FID sweep の推論パイプライン実装

### 確定方針
- サンプラー:Euler(Heun2 は配置差をならすため不採用)
- NFE=50、FID サンプル数=50K、生成バッチ=250、seed=全セル共通
- 失敗時:fid.json があれば skip して再開可能にする

### 出力フォーマット(fid.json)
```json
{
  "train_dist": "uniform", "sampling_dist": "center",
  "fid": 5.123, "ode_method": "euler", "nfe": 50,
  "fid_samples": 50000, "batch_size": 250, "seed": 0,
  "elapsed_sec": 400, "timestamp": "2026-..."
}
```

## 禁止事項

- /home/jovyan/work/srv21/ 外にファイルを作らない
- output_<dist>/checkpoint.pth を触らない
- git push を確認なしにしない
- training/ の既存コードを大幅にリファクタしない
