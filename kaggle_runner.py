"""
Утилита для запуска пайплайна из main.py в Kaggle Notebook/Script.

Типовой сценарий в ноутбуке:
```
%%bash
# при необходимости поставить timm (torch/torchvision уже предустановлены)
pip install -q timm==1.0.22
```

Запуск обучения (сохранит чекпойнты в /kaggle/working/outputs):
```
!python kaggle_runner.py --mode train --epochs 10 --image-size 320 --batch-size 16
```

Инференс и формирование submission.csv:
```
!python kaggle_runner.py --mode infer --checkpoint-dir /kaggle/working/outputs --submission /kaggle/working/submission.csv
```
"""

import argparse
import os

from main import (
    load_train_dataframe,
    load_test_dataframe,
    run_training,
    run_inference,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CSIRO Kaggle runner")
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument(
        "--data-dir",
        default="/kaggle/input/csiro-biomass",
        help="Путь к данным соревнования в среде Kaggle",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/outputs",
        help="Куда сохранять чекпойнты (train)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/kaggle/working/outputs",
        help="Откуда брать чекпойнты (infer)",
    )
    parser.add_argument(
        "--submission",
        default="/kaggle/working/submission.csv",
        help="Путь для сохранения сабмита (infer)",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument(
        "--no-pretrain", action="store_true", help="Не использовать предобученные веса"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Использовать mixed precision (если CUDA)"
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--fold", type=int, default=None, help="Если указан — обучаем только этот фолд"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tta",
        type=int,
        default=1,
        help="Число TTA трансформов при инференсе (1 или 2)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        os.makedirs(args.output_dir, exist_ok=True)
        train_df, state2idx, species2idx, num_stats = load_train_dataframe(
            args.data_dir
        )
        run_training(train_df, state2idx, species2idx, num_stats, args)
    else:
        # Для инференса нам нужно знать state/species словари; берем их из train
        train_df, state2idx, species2idx, num_stats = load_train_dataframe(
            args.data_dir
        )
        test_df = load_test_dataframe(args.data_dir, state2idx, species2idx, num_stats)
        run_inference(test_df, state2idx, species2idx, args)


if __name__ == "__main__":
    main()
