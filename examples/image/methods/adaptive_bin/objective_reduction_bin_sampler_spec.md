現在の flow_matching リポジトリに、Flow Matching 用の objective-reduction adaptive bin sampler を新規実装してください。

目的:
Kim et al. “Adaptive Non-uniform Timestep Sampling for Accelerating Diffusion Model Training” の考え方に基づき、raw per-bin loss の大小ではなく、「その bin で更新した結果、全体の Flow Matching objective がどれだけ下がったか」を使って timestep/bin sampling probability を更新する。

重要:
- 基本的に新規ファイルのみで実装してください。
- 既存の train_phase1_twophase.py、train_phase1_bin_adaptive.py、train_phase1_bin_loss_aware.py、既存 sampler 実装は変更しないでください。
- 既存実験、既存 checkpoint、既存 W&B logging に影響を与えないでください。
- 既存ファイルの変更がどうしても必要な場合は、最小限にし、理由を明確に報告してください。

新規ファイル:
1. examples/image/objective_reduction_bin_sampler.py
2. examples/image/train_phase1_bin_objective_reduction.py

train_phase1_bin_objective_reduction.py は、既存の train_phase1_bin_adaptive.py または train_phase1_bin_loss_aware.py を参考にしつつ、新規スクリプトとして作成してください。

============================================================
1. 手法の概要
============================================================

Flow Matching の訓練時刻 t ∈ [0,1] を K 個の bin に分ける。

K=10 の場合:
bin 0: [0.0, 0.1)
bin 1: [0.1, 0.2)
...
bin 9: [0.9, 1.0]

1 batch につき bin を1つ選び、その bin の範囲内で batch size 分の t を一様サンプリングする。
1-batch-1-bin 方式を維持する。

この sampler は raw per-bin loss の大小を使わない。
代わりに、選ばれた bin で UNet を1 step 更新した前後で、評価時刻集合 S 上の平均 FM loss がどれだけ下がったかを reward とする。

評価時刻 S は K 個の bin center とする。

K=10 の場合:
S = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

評価 objective:
L_S(theta) = (1/K) * sum_j L_FM(theta, tau_j)

reward は相対改善:
reward = (eval_loss_before - eval_loss_after) / (eval_loss_before + 1e-8)

reward > 0 なら、その bin での更新により評価時刻全体の FM loss が下がったことを意味する。

============================================================
2. ObjectiveReductionBinSampler の仕様
============================================================

examples/image/objective_reduction_bin_sampler.py に ObjectiveReductionBinSampler を実装してください。

デフォルトパラメータ:
- k = 10
- reward_ema_beta = 0.9
- temperature = 1.0
- uniform_mix = 0.1
- update_sampler_every = 40
- warmup_steps = 1000
- importance_gamma = 0.5
- eps = 1e-8

保持する状態:
- reward_ema: shape [K] tensor, 初期値 zeros
- last_probs: shape [K] tensor, 初期値 uniform
- selected_bin: int or None
- selected_prob: float or tensor
- num_sampler_updates: int
- round_robin_idx: int
- last_reward: float
- last_eval_loss_before: float
- last_eval_loss_after: float
- last_importance_weight: float

optimizer は不要です。
bin_logits も不要です。
この sampler は gradient-based parameter を持たず、reward_ema から直接 probability を作ります。

============================================================
3. Probability の作り方
============================================================

reward_ema から sampling probability を作る。

p_reward = softmax(reward_ema / temperature)

探索を残すため、一様分布を混ぜる。

probs = (1 - uniform_mix) * p_reward + uniform_mix / K

probs.sum() は 1 になること。
uniform_mix により、各 bin の確率は最低でも uniform_mix / K 程度になる。

warmup 中、つまり global_step < warmup_steps では、sampling は完全一様にする。
ただし reward_ema の更新は warmup 中も行う。

============================================================
4. Sampling
============================================================

sample_t(batch_size, device, global_step, force_bin=None) のような API にしてください。

通常 step:
- global_step < warmup_steps なら、一様確率で bin を選ぶ
- warmup 後は probs に従って categorical sampling する

sampler update step:
- round-robin forced exploration を使う
- sampler update step では、force_bin または sampler 内部の round_robin_idx により bin を強制選択する
- round-robin は以下の順番:
  0 -> 1 -> 2 -> ... -> K-1 -> 0 -> ...
- sampler update step では、training に使った bin と reward 更新対象 bin を一致させること

選ばれた bin b の範囲:
[t_low, t_high) = [b/K, (b+1)/K)

その範囲で batch size 分の t を一様サンプリング:
t = t_low + rand(batch_size) / K

t は clamp(1e-6, 1 - 1e-6) してよい。

sample_t は以下を内部状態に保存する:
- selected_bin
- selected_prob
- selected bin に対する current probability

============================================================
5. Evaluation function
============================================================

objective_reduction_bin_sampler.py に、評価用の補助関数を作ってください。

例:
eval_fm_loss_at_bin_centers(samples, noise, path, model, k)

仕様:
- torch.no_grad() で実行
- S = bin centers = (i + 0.5) / k, i=0..K-1
- 各 tau について:
  t_eval = torch.full((bs,), tau, device=samples.device)
  ps = path.sample(t=t_eval, x_0=noise, x_1=samples)
  pred = model(ps.x_t, t_eval, extra={})  # repo の forward API に合わせる
  loss = (pred - ps.dx_t).pow(2).mean()
- 各 tau の loss を tensor [K] として返す
- 平均 lossも呼び出し側で計算できるようにする

reward eval は raw UNet を使う。
EMA wrapper ではなく raw_model を渡してください。
ただし通常の training forward は既存スクリプトと同じ挙動を維持してください。

============================================================
6. Sampler update
============================================================

sampler update は update_sampler_every step ごとに行う。
update 判定は既存コードの流儀に合わせてよいが、基本は:
do_sampler_update = (global_step + 1) % update_sampler_every == 0

sampler update step では:
1. round-robin bin を強制選択して、その bin で training t をサンプルする
2. UNet 更新前に eval_loss_before を計算する
3. 通常の FM training loss で UNet を更新する
4. optimizer.step()
5. scheduler.step()
6. ema_model.update_ema()
7. UNet 更新後に eval_loss_after を計算する
8. relative reward を計算する
   reward = (eval_loss_before - eval_loss_after) / (eval_loss_before + eps)
9. 選ばれた bin の reward_ema を更新する
   reward_ema[b] = reward_ema_beta * reward_ema[b] + (1 - reward_ema_beta) * reward
10. reward_ema から last_probs を更新する
11. round_robin_idx を次に進める
12. num_sampler_updates += 1

選ばれていない bin の reward_ema は変更しない。

============================================================
7. Training loss と partial importance weighting
============================================================

training loss には partial importance weighting を入れる。

base_loss:
base_loss = (pred - ps.dx_t).pow(2).mean()

選ばれた bin b の probability を q_b とする。
importance weight:
importance_weight = (1 / (K * q_b)) ** importance_gamma

weighted loss:
loss = importance_weight * base_loss

importance_gamma = 0.5 をデフォルトにする。

注意:
- reward eval loss には importance weight を掛けない。
- importance weighting は training loss にのみ適用する。
- warmup 中に一様 sampling の場合は q_b = 1/K なので importance_weight = 1 になるはず。

ログ用に base_loss と weighted loss の両方を保持・出力してください。

============================================================
8. train_phase1_bin_objective_reduction.py
============================================================

新規スクリプトを作成してください。

既存 phase1 系スクリプトと同じ基本構造を維持:
- CIFAR-10 dataset
- UNet + EMA
- optimizer
- scheduler
- checkpoint
- FID eval
- per-timestep loss
- snapshot
- dry-run
- resume
- W&B

ただし checkpoint dir は分ける:
checkpoints/phase1_bin_objective_reduction/

W&B project は既存 adaptive-bin と同じでもよいが、config.mode で区別する:
mode = "objective_reduction_bin"

sampler_id default は主要パラメータを含める:
objred_K10_T1.0_mix0.1_beta0.9_g0.5

CLI 引数:
- --bin-k default 10
- --reward-ema-beta default 0.9
- --temperature default 1.0
- --uniform-mix default 0.1
- --update-sampler-every default 40
- --warmup-steps default 1000
- --importance-gamma default 0.5
- 既存の run 管理引数:
  --sampler-id
  --dry-run
  --max-epochs
  --no-resume
  --resume-from
  --keep-epochs
  --ckpt-every
  --keep-recent-n
  --keep-all
  --eval-every
  --snapshot-every
  --fid-samples
  --data-path
  --num-workers
  --device
  --seed
  --eval-only
などは train_phase1_bin_adaptive.py / train_phase1_bin_loss_aware.py と同様に維持する。

train_one_epoch の signature:
train_one_epoch(ema_model, raw_model, dataloader, optimizer, scheduler,
                device, epoch, global_step, run, bin_sampler)

raw_model は reward eval に使う。
training forward は既存の ema_model(...) の流儀を維持してよい。

============================================================
9. W&B logging
============================================================

sampler update step でログ:
- sampler/bin_prob_0 ... sampler/bin_prob_9
- sampler/reward_ema_0 ... sampler/reward_ema_9
- sampler/selected_bin
- sampler/q_selected
- sampler/reward
- sampler/relative_reward
- sampler/eval_loss_before
- sampler/eval_loss_after
- sampler/eval_loss_mean_before
- sampler/eval_loss_mean_after
- sampler/importance_weight
- sampler/base_train_loss
- sampler/weighted_train_loss
- sampler/entropy
- sampler/max_prob
- sampler/min_prob
- sampler/update_flag = 1
- sampler/round_robin_idx
- sampler/num_sampler_updates
- global_step

通常 100 step ログ:
- train/loss
- train/base_loss
- train/lr
- sampler/bin_prob_0 ... sampler/bin_prob_9
- sampler/reward_ema_0 ... sampler/reward_ema_9
- sampler/selected_bin
- sampler/q_selected
- sampler/importance_weight
- sampler/entropy
- sampler/max_prob
- sampler/min_prob
- sampler/update_flag = 0
- global_step

epoch/eval/FID logging は既存 phase1 系スクリプトと同様に維持。

W&B config:
- mode = objective_reduction_bin
- bin_k
- reward_ema_beta
- temperature
- uniform_mix
- update_sampler_every
- warmup_steps
- importance_gamma
- eval_times = bin centers
- checkpoint dir
- batch size, lr, ema_decay など既存設定

可能なら wandb.define_metric("*", step_metric="global_step") を使い、W&B の内部 Step と training global_step が混ざらないようにしてください。
既存実験を壊さない範囲で、この新規スクリプト内だけで設定してください。

============================================================
10. Checkpoint
============================================================

ObjectiveReductionBinSampler に state_dict/load_state_dict を実装してください。

保存対象:
- type = "objective_reduction_bin"
- k
- reward_ema
- last_probs
- selected_bin
- selected_prob
- num_sampler_updates
- round_robin_idx
- last_reward
- last_eval_loss_before
- last_eval_loss_after
- last_importance_weight
- hyperparameters:
  reward_ema_beta
  temperature
  uniform_mix
  update_sampler_every
  warmup_steps
  importance_gamma

load_state_dict:
- sampler_state が無い場合は skip
- type が違う場合は warning を出して skip
- k が違う場合は warning を出して skip
- tensor は現在 device に移す
- 既存 checkpoint との互換性を壊さないこと

save_checkpoint には sampler_state=bin_sampler.state_dict() を渡してください。
既存 checkpoint helper が sampler_state に対応している前提で使ってください。
もし対応が必要な場合も、既存実験を壊さない optional field として最小変更にしてください。

============================================================
11. 検証
============================================================

実装後、以下を確認してください。

1. python compile / syntax check
   - objective_reduction_bin_sampler.py
   - train_phase1_bin_objective_reduction.py

2. sampler 単体テスト相当
   - probs.sum() == 1
   - warmup 中は一様 probability
   - uniform_mix により min prob >= uniform_mix/K 程度
   - round-robin が 0,1,2,...,K-1,0... と進む
   - selected bin の t が正しい範囲に入る
   - importance_weight が q_b=1/K のとき 1 になる
   - state_dict/load_state_dict が round-trip する
   - type mismatch / k mismatch で crash しない

3. 小モデル・合成データなどで train_one_epoch の smoke test
   - sampler update step で round-robin bin が使われる
   - eval before/after が計算される
   - reward_ema が選ばれた bin だけ更新される
   - optimizer.step -> scheduler.step -> ema_model.update_ema の順序が維持される
   - W&B/fake run payload に必須キーが含まれる
   - reward/relative_reward, importance_weight, base_loss, weighted_loss がログされる

4. 可能なら --dry-run
   - ただし本物の CIFAR-10 download、FID、wandb login 等で重い/失敗する場合は、実行しなかった理由または失敗理由を報告してください。

実装後の報告:
- 追加/変更ファイル
- 既存ファイルを変更したかどうか
- 主要な設計判断
- 検証結果
- dry-run 実行可否
を簡潔にまとめてください。