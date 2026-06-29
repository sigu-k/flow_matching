# &lt;method_name&gt;

手法の概要を1〜2行で。何を変えた手法か(t の分布 / サンプラー / 損失など)。

## 使い方

```bash
# methods/_template/ をコピーして手法名にリネーム
cp -r methods/_template methods/<method_name>

# 実行(ログは logs/<method_name>/ に自動で残る)
bash methods/<method_name>/run.sh --<args>
```

## 出力

- チェックポイント: `checkpoints/<method_name>/`(リポジトリルート直下)
- ログ: `examples/image/logs/<method_name>/`

## メモ

- 共有資産(`phase1_utils` / `train_arg_parser` / `models` / `training`)は
  `examples/image` 直下のものを参照する。`train.py` の import ヘッダが
  `image/` を `sys.path` に通すので、cwd やサーバに依存せず動く。
- 既存手法のスクリプト(`image/` 直下の `train_phase1_*.py`)はこの規約の対象外。
