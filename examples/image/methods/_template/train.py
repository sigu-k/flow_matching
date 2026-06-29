"""新規手法の学習エントリ雛形。

このファイルを methods/<method_name>/train.py にコピーし、
プレースホルダ <method_name> を実際の手法名に置き換えて使う。

共有資産(phase1_utils / train_arg_parser / models / training)は
examples/image 直下にあるため、下の import ヘッダで image/ を sys.path に通す。
サーバ名は一切埋め込まず、すべて __file__ 基準で導出すること
(CLAUDE.md のポータビリティルール準拠)。
"""

import sys
from pathlib import Path

# --- import ヘッダ(methods/<method>/train.py 用、parents の数は階層に依存)---
# image/methods/<method>/train.py の場合: parents[2]=image, parents[4]=repo root
_IMAGE_DIR = Path(__file__).resolve().parents[2]   # examples/image
_REPO_ROOT = Path(__file__).resolve().parents[4]   # flow_matching (repo root)
if str(_IMAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_IMAGE_DIR))            # phase1_utils, models, training を解決

# 出力先(手法名で固定する)
METHOD_NAME = Path(__file__).resolve().parent.name  # = ディレクトリ名
CKPT_BASE = _REPO_ROOT / "checkpoints" / METHOD_NAME
LOG_DIR = _IMAGE_DIR / "logs" / METHOD_NAME

# --- ここから先は共有資産を普通に import できる ---
# from phase1_utils import ...
# from models.unet import UNetModel
# from train_arg_parser import get_args_parser


def main() -> None:
    CKPT_BASE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        f"{METHOD_NAME}: train.py を実装してください "
        f"(ckpt={CKPT_BASE}, log={LOG_DIR})"
    )


if __name__ == "__main__":
    main()
