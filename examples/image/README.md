# Image example

## Training instructions

1. Download and unpack blurred ImageNet from the [official website](https://image-net.org/download.php).

```
export IMAGENET_DIR=~/flow_matching/examples/image/data/
export IMAGENET_RES=64
tar -xf ~/Downloads/train_blurred.tar.gz -C $IMAGENET_DIR
```

2. Downsample Imagenet to the desired resolution.

```
cd ~/
git clone git@github.com:PatrykChrabaszcz/Imagenet32_Scripts.git
python Imagenet32_Scripts/image_resizer_imagent.py -i ${IMAGENET_DIR}train_blurred -o ${IMAGENET_DIR}train_blurred_$IMAGENET_RES -s $IMAGENET_RES -a box  -r -j 10 
```

3. Set up the virtual environment. First, set up the virtual environment by following the steps in the repository's `README.md`. Then,

```
conda activate flow_matching

cd examples/image
pip install -r requirements.txt
```

4. [Optional] Test-run training locally. A test run executes one step of training followed by one step of evaluation.

```
python train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ --test_run
```

5. Launch training on a SLURM cluster

```
python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ 
```

6. Evaluate the model using the `--eval_only` flag. The evaluation script will generate snapshots under the `/snapshots` folder. Specify the `--compute_fid` flag to also compute the FID with respect to the training set. Make sure to specify your most recent checkpoint to resume from. The results are printed to `log.txt`.

```
python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ --resume=./output_dir/checkpoint-899.pth --compute_fid --eval_only
```


## Results
| Data                  | Model type                       | Epochs | FID  | Command                                                                                                                                                                                                                                                                                                                                                   |
|-----------------------|----------------------------------|-------|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cifar10               | Unconditional UNet               | 1800  | 2.07 | `python submitit_train.py \`<br>`--dataset=cifar10 \`<br>`--batch_size=64 \`<br>`--nodes=1 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--epochs=3000 \`<br>`--class_drop_prob=1.0 \`<br>`--cfg_scale=0.0 \`<br>`--compute_fid \`<br>`--ode_method heun2 \`<br>`--ode_options '{"nfe": 50}' \`<br>`--use_ema \`<br>`--edm_schedule \`<br>`--skewed_timesteps` |
| ImageNet32 (Blurred)  | Class conditional Unet           | 900   | 1.14 | `export IMAGENET_RES=32 \`<br>`python submitit_train.py \`<br>`--data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ \`<br>`--batch_size=32 \`<br>`--nodes=8 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--decay_lr \`<br>`--compute_fid \`<br>`--ode_method dopri5 \`<br>`--ode_options '{"atol": 1e-5, "rtol":1e-5}'` |
| ImageNet64 (Blurred)  | Class conditional Unet           | 900   | 1.64 | `export IMAGENET_RES=64 \`<br>`python submitit_train.py \`<br>`--data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ \`<br>`--batch_size=32 \`<br>`--nodes=8 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--decay_lr \`<br>`--compute_fid \`<br>`--ode_method dopri5 \`<br>`--ode_options '{"atol": 1e-5, "rtol":1e-5}'` |
| Cifar10 (Discrete Flow) | Unconditional Unet           | 2500   | 3.58 | `python submitit_train.py \`<br>`--dataset=cifar10 \`<br>`--nodes=1 \`<br>`--discrete_flow_matching \`<br>`--batch_size=32 \`<br>`--accum_iter=1 \`<br>`--cfg_scale=0.0 \`<br>`--use_ema \`<br>`--epochs=3000 \`<br>`--class_drop_prob=1.0 \`<br>`--compute_fid \`<br>`--sym_func` |



---

## Phase 0: Uniform Timestep Saturation Run (300 epochs)

**目的**: Curriculum timestep sampling 本実験の前段階として、Uniform t ~ U(0,1) で 300 epoch 訓練し、FID@50k が saturate する epoch を特定する。

### 前提セットアップ

```bash
source ~/work/srv11/setup_env.sh
cd /home/jovyan/work/srv11/flow_matching/examples/image
```

### Dry-run（動作確認・本番前に必須）

2 epoch だけ回して train loop・wandb・FID・checkpoint・resume が全て正常かを確認する。

```bash
# ステップ1: 2 epoch 実行
python train_phase0_uniform.py --dry-run

# 確認事項:
# - wandb に train/loss が流れる
# - epoch 2 末で FID 評価が動く (EMA + raw 両方)
# - ~/work/srv11/checkpoints/phase0/ckpt_epoch002.pt が作成される
# - ~/work/srv11/checkpoints/phase0/latest.pt が作成される
# - ~/work/srv11/checkpoints/phase0/fid_history.json が作成される
```

```bash
# ステップ2: Resume テスト（同じコマンドを再実行）
python train_phase0_uniform.py --dry-run

# 確認事項:
# - ログに "Resuming from epoch 2, global_step ..., wandb run_id=..." が出る
# - wandb の同じ run が継続更新される
```

```bash
# ステップ3: GPU メモリ確認（別ターミナルで）
nvidia-smi
# 期待値: < 20 GB
```

### 本番 run（300 epoch）

dry-run が全て通った後に実行:

```bash
# 新規 run
python train_phase0_uniform.py

# または nohup でバックグラウンド実行
nohup python train_phase0_uniform.py > ~/work/srv11/checkpoints/phase0/train.log 2>&1 &
```

### Resume（container reset 後）

container が落ちた場合、同じコマンドを再実行するだけで自動 resume される:

```bash
python train_phase0_uniform.py
# → 自動的に latest.pt から再開
```

特定の checkpoint から再開する場合:

```bash
python train_phase0_uniform.py --resume-from ~/work/srv11/checkpoints/phase0/ckpt_epoch100.pt
```

強制的に最初から始める場合:

```bash
python train_phase0_uniform.py --no-resume
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--dry-run` | false | 2 epoch・eval_every=2 で smoke-test |
| `--max-epochs N` | 300 | エポック数を上書き |
| `--eval-every N` | 20 | FID 評価頻度 (epoch 単位) |
| `--no-resume` | false | 既存 checkpoint を無視して最初から |
| `--resume-from PATH` | - | 特定 checkpoint から再開 |
| `--data-path PATH` | `./data/image_generation` | CIFAR-10 データディレクトリ |
| `--num-workers N` | 4 | DataLoader ワーカー数 |
| `--device` | `cuda` | 使用デバイス |

### 出力ファイル

| ファイル | 更新タイミング | 内容 |
|---|---|---|
| `checkpoints/phase0/latest.pt` | 毎 epoch | 最新 checkpoint（resume 用）|
| `checkpoints/phase0/ckpt_epoch{NNN}.pt` | 20 epoch ごと | 定期 checkpoint |
| `checkpoints/phase0/fid_history.json` | 20 epoch ごと | FID 履歴 |
| wandb `phase0-uniform-saturation` | 毎 100 step / 20 epoch | loss・LR・FID・per-timestep loss |

### 固定ハイパーパラメータ

- UNet: model_channels=128, channel_mult=[2,2,2], num_res_blocks=4, attention_resolutions=[2], num_heads=1, dropout=0.3, use_scale_shift_norm=True (~111M params)
- Batch size: 64, AdamW β=(0.9, 0.95), Peak LR=1e-4
- LR schedule: linear warmup 10k steps → constant
- EMA decay: 0.99995
- Timestep: t ~ Uniform(0, 1)
- FID: 50k samples, 50-NFE Euler, batch 250, seed=0, fp32

---

## ディレクトリ規約(新規手法)

今後新しく追加する手法は、`image/` 直下にファイルを散らさず、手法ごとに
`methods/<method_name>/` ディレクトリへまとめる。

```
image/methods/<method_name>/
├── train.py          # 学習エントリ
├── run.sh            # 実行スクリプト(ログ先も固定)
├── <method>_*.py     # その手法固有のモジュール(あれば)
└── README.md         # 手法の概要・実行手順
```

- **雛形**: `methods/_template/` をコピーして始める(`cp -r methods/_template methods/<method_name>`)。
- **ログ/出力**: `image/logs/<method_name>/` に固定。`run.sh` が自動で作成・追記する。
- **チェックポイント**: `checkpoints/<method_name>/`(リポジトリルート直下)。
- **共有資産**: `phase1_utils.py` / `train_arg_parser.py` / `models/` / `training/` は
  `image/` 直下のまま据え置き。新規手法の `train.py` は import ヘッダで `image/` を
  `sys.path` に通して参照する(サーバ名・cwd 非依存、`__file__` 基準で導出)。

```python
# methods/<method>/train.py の import ヘッダ
import sys
from pathlib import Path

_IMAGE_DIR = Path(__file__).resolve().parents[2]   # examples/image
_REPO_ROOT = Path(__file__).resolve().parents[4]   # flow_matching
if str(_IMAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_IMAGE_DIR))            # phase1_utils, models, training を解決
```

**既存の `image/` 直下スクリプト(`train_phase1_*.py` など)はこの規約の対象外**で、
当面そのまま据え置く(import 前提が変わるため一括移行はしない)。

---

## Acknowledgements

This example partially use code from:
- [Guided diffusion](https://github.com/openai/guided-diffusion/)
- [ConvNext](https://github.com/facebookresearch/ConvNeXt)

## License

The majority of the code in this example is licensed under CC-BY-NC, however portions of the project are available under separate license terms: 
- The UNet model is under MIT license.
- The distributed computing and the grad scaler code is under MIT license.

## Citations

Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.

Karras, Tero, et al. "Elucidating the design space of diffusion-based generative models." Advances in neural information processing systems 35 (2022): 26565-26577.

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer International Publishing, 2015.
