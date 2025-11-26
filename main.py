"""
CSIRO Image2Biomass: базовый пайплайн обучения/инференса.

Функционал:
- Чтение train/test CSV, сводка таргетов по изображению.
- Простая модель: CNN-бекбон (timm/torchvision) + табличные фичи (категории -> эмбеддинги).
- Взвешенный MSE по таргетам и метрика взвешенного R² (как в соревновании).
- K-fold обучение, сохранение лучших чекпойнтов.
- Инференс с усреднением по фолдам и простой TTA (flip).

Пример запуска:
  Обучение всех фолдов:
    python main.py --mode train --data-dir data --output-dir outputs --epochs 15 --image-size 384 --backbone convnext_tiny
  Инференс (ищет чекпойнты outputs/fold_*.pt):
    python main.py --mode infer --data-dir data --checkpoint-dir outputs --submission submission.csv --image-size 384 --backbone convnext_tiny
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Попробуем timm, fallback на torchvision, если timm нет
try:
    import timm
except ImportError:  # pragma: no cover - timm может отсутствовать локально
    timm = None
from torchvision import models, transforms

# Порядок таргетов фиксируем для согласованности с весами/метрикой
TARGET_ORDER = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}


# =========================
# Утилиты
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Простое разбиение на фолды (перемешиваем и раскладываем по модулю)."""
    idx = np.arange(len(df))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    fold_ids = np.zeros(len(df), dtype=int)
    for i, j in enumerate(idx):
        fold_ids[j] = i % n_folds
    df = df.copy()
    df["fold"] = fold_ids
    return df


def parse_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Разбор даты формата 'YYYY/M/D' -> year, month, day; если нет колонки, заполняем нулями."""
    if col not in df.columns:
        df["year"] = 0
        df["month"] = 0
        df["day"] = 0
        return df
    dt = pd.to_datetime(df[col], errors="coerce")
    df["year"] = dt.dt.year.fillna(0).astype(int)
    df["month"] = dt.dt.month.fillna(0).astype(int)
    df["day"] = dt.dt.day.fillna(0).astype(int)
    return df


def build_target_array(row: pd.Series) -> np.ndarray:
    """Строим вектор таргетов по TARGET_ORDER."""
    return row[TARGET_ORDER].to_numpy(dtype=np.float32)


def compute_num_stats(
    df: pd.DataFrame, cols: List[str]
) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for c in cols:
        mean = df[c].mean() if c in df.columns else 0.0
        std = df[c].std() if c in df.columns else 1.0
        stats[c] = (mean, std if std > 1e-6 else 1.0)
    return stats


def standardize_row(
    row: pd.Series, stats: Dict[str, Tuple[float, float]], cols: List[str]
) -> List[float]:
    vals = []
    for c in cols:
        mean, std = stats.get(c, (0.0, 1.0))
        x = row[c] if c in row and not pd.isna(row[c]) else 0.0
        vals.append((float(x) - mean) / std)
    return vals


def extract_image_id(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


# =========================
# Данные
# =========================
def load_train_dataframe(
    data_dir: str,
) -> Tuple[
    pd.DataFrame, Dict[str, int], Dict[str, int], Dict[str, Tuple[float, float]]
]:
    """Читает train.csv, сворачивает в wide-формат по таргетам, кодирует категории."""
    train_path = os.path.join(data_dir, "train.csv")
    df = pd.read_csv(train_path)
    df["image_id"] = df["sample_id"].str.split("__").str[0]
    df = parse_date(df, "Sampling_Date")

    # агрегируем метаданные по изображению
    meta_cols = [
        "image_path",
        "image_id",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
        "year",
        "month",
        "day",
    ]
    
    meta = df[meta_cols].drop_duplicates("image_path")

    # pivot таргеты
    pivot = df.pivot(
        index="image_path", columns="target_name", values="target"
    ).reset_index()
    train_df = meta.merge(pivot, on="image_path", how="inner")
    train_df["image_id"] = train_df["image_id"].fillna(
        train_df["image_path"].apply(extract_image_id)
    )

    # кодирование категорий
    state_values = sorted(train_df["State"].dropna().unique().tolist())
    species_values = sorted(train_df["Species"].dropna().unique().tolist())
    state2idx = {v: i for i, v in enumerate(state_values)}
    species2idx = {v: i for i, v in enumerate(species_values)}
    train_df["state_idx"] = train_df["State"].map(state2idx).fillna(0).astype(int)
    train_df["species_idx"] = train_df["Species"].map(species2idx).fillna(0).astype(int)

    # численные фичи (заполняем нулями при отсутствии)
    num_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm", "year", "month", "day"]
    for c in num_cols:
        if c not in train_df.columns:
            train_df[c] = 0.0
        train_df[c] = train_df[c].fillna(0.0)
    num_stats = compute_num_stats(train_df, num_cols)
    train_df["num_feats"] = train_df.apply(
        lambda r: standardize_row(r, num_stats, num_cols), axis=1
    )

    # вектор таргетов
    train_df["targets"] = train_df.apply(build_target_array, axis=1)

    return train_df, state2idx, species2idx, num_stats


def load_test_dataframe(
    data_dir: str,
    state2idx: Dict[str, int],
    species2idx: Dict[str, int],
    num_stats: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """Читает test.csv и формирует по одному ряду на изображение (без таргетов)."""
    test_path = os.path.join(data_dir, "test.csv")
    df = pd.read_csv(test_path)
    df["image_id"] = df["sample_id"].str.split("__").str[0]
    df_unique = df[["image_path", "image_id"]].drop_duplicates("image_path").copy()

    # метаданные могут отсутствовать, поэтому ставим заглушки
    df_unique["State"] = "unknown"
    df_unique["Species"] = "unknown"
    df_unique["Sampling_Date"] = ""
    df_unique = parse_date(df_unique, "Sampling_Date")
    df_unique["state_idx"] = df_unique["State"].map(state2idx).fillna(0).astype(int)
    df_unique["species_idx"] = (
        df_unique["Species"].map(species2idx).fillna(0).astype(int)
    )

    num_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm", "year", "month", "day"]
    for c in num_cols:
        df_unique[c] = 0.0
    df_unique["num_feats"] = df_unique.apply(
        lambda r: standardize_row(r, num_stats, num_cols), axis=1
    )
    return df_unique


class PastureDataset(Dataset):
    """Датасет: изображение + табличные фичи (+ таргет в train режиме)."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        transforms_compose: transforms.Compose,
        is_train: bool,
    ) -> None:
        self.paths = df["image_path"].tolist()
        self.image_ids = (
            df["image_id"].tolist()
            if "image_id" in df.columns
            else [extract_image_id(p) for p in self.paths]
        )
        self.targets = df["targets"].tolist() if is_train else None
        self.state_idx = (
            df["state_idx"].tolist()
            if "state_idx" in df.columns
            else [0] * len(self.paths)
        )
        self.species_idx = (
            df["species_idx"].tolist()
            if "species_idx" in df.columns
            else [0] * len(self.paths)
        )
        self.num_feats = (
            df["num_feats"].tolist()
            if "num_feats" in df.columns
            else [[0.0] * 5 for _ in self.paths]
        )
        self.data_dir = data_dir
        self.is_train = is_train
        self.transforms = transforms_compose

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.data_dir, self.paths[idx])
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        img = self.transforms(img)
        sample = {
            "image": img,
            "state_idx": torch.tensor(self.state_idx[idx], dtype=torch.long),
            "species_idx": torch.tensor(self.species_idx[idx], dtype=torch.long),
            "num_feats": torch.tensor(self.num_feats[idx], dtype=torch.float32),
            "image_id": self.image_ids[idx],
        }
        if self.is_train and self.targets is not None:
            sample["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sample


# =========================
# Модель
# =========================
class PastureModel(nn.Module):
    def __init__(
        self,
        num_states: int,
        num_species: int,
        num_numeric: int,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = self._create_backbone(backbone_name, pretrained)

        # эмбеддинги категорий
        self.state_emb = nn.Embedding(num_states if num_states > 0 else 1, 16)
        self.species_emb = nn.Embedding(num_species if num_species > 0 else 1, 32)

        head_in = feat_dim + 16 + 32 + num_numeric
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, len(TARGET_ORDER)),
        )

    def _create_backbone(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        if timm is not None:
            model = timm.create_model(
                name, pretrained=pretrained, num_classes=0, global_pool="avg"
            )
            feat_dim = model.num_features
            return model, feat_dim
        # fallback на torchvision resnet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    def forward(
        self,
        images: torch.Tensor,
        state_idx: torch.Tensor,
        species_idx: torch.Tensor,
        num_feats: torch.Tensor,
    ) -> torch.Tensor:
        img_feat = self.backbone(images)
        state_feat = self.state_emb(state_idx)
        species_feat = self.species_emb(species_idx)
        concat = torch.cat([img_feat, state_feat, species_feat, num_feats], dim=1)
        return self.head(concat)


# =========================
# Лоссы и метрики
# =========================
class WeightedMSELoss(nn.Module):
    def __init__(self, weights: Dict[str, float]) -> None:
        super().__init__()
        self.register_buffer(
            "weights",
            torch.tensor([weights[t] for t in TARGET_ORDER], dtype=torch.float32),
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds/targets: [B, 5]
        w = self.weights.to(preds.device)
        return torch.mean(w * (preds - targets) ** 2)


def weighted_r2_score(
    preds: torch.Tensor, targets: torch.Tensor, weights: Dict[str, float]
) -> float:
    w = torch.tensor(
        [weights[t] for t in TARGET_ORDER], device=preds.device, dtype=torch.float32
    )
    w = w.unsqueeze(0).expand_as(preds)
    y = targets
    y_hat = preds
    w_sum = w.sum()
    y_mean = torch.sum(w * y) / w_sum
    numerator = torch.sum(w * (y - y_hat) ** 2)
    denominator = torch.sum(w * (y - y_mean) ** 2) + 1e-8
    r2 = 1.0 - numerator / denominator
    return r2.item()


# =========================
# Трансформации
# =========================
def build_transforms(image_size: int, mode: str) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


# =========================
# Обучение
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        state_idx = batch["state_idx"].to(device)
        species_idx = batch["species_idx"].to(device)
        num_feats = batch["num_feats"].to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(images, state_idx, species_idx, num_feats)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images, state_idx, species_idx, num_feats)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            state_idx = batch["state_idx"].to(device)
            species_idx = batch["species_idx"].to(device)
            num_feats = batch["num_feats"].to(device)

            preds = model(images, state_idx, species_idx, num_feats)
            loss = criterion(preds, targets)
            total_loss += loss.item() * images.size(0)
            all_preds.append(preds)
            all_targets.append(targets)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = weighted_r2_score(all_preds, all_targets, TARGET_WEIGHTS)
    return total_loss / len(loader.dataset), r2


def run_training(
    train_df: pd.DataFrame,
    state2idx: Dict[str, int],
    species2idx: Dict[str, int],
    num_stats: Dict[str, Tuple[float, float]],
    args,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = make_folds(train_df, args.n_folds, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for fold in range(args.n_folds):
        if args.fold is not None and fold != args.fold:
            continue
        print(f"\n=== Fold {fold} / {args.n_folds} ===")
        train_split = train_df[train_df["fold"] != fold].reset_index(drop=True)
        val_split = train_df[train_df["fold"] == fold].reset_index(drop=True)

        train_ds = PastureDataset(
            train_split,
            args.data_dir,
            build_transforms(args.image_size, "train"),
            is_train=True,
        )
        val_ds = PastureDataset(
            val_split,
            args.data_dir,
            build_transforms(args.image_size, "val"),
            is_train=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = PastureModel(
            num_states=max(len(state2idx), 1),
            num_species=max(len(species2idx), 1),
            num_numeric=len(train_ds[0]["num_feats"]),
            backbone_name=args.backbone,
            pretrained=not args.no_pretrain,
            hidden_dim=args.hidden_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        criterion = WeightedMSELoss(TARGET_WEIGHTS).to(device)

        scaler = (
            torch.cuda.amp.GradScaler()
            if args.amp and torch.cuda.is_available()
            else None
        )

        best_r2 = -1e9
        best_path = os.path.join(args.output_dir, f"fold_{fold}.pt")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_loss, val_r2 = validate(model, val_loader, criterion, device)
            dt = time.time() - t0

            print(
                f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_r2 {val_r2:.4f} | {dt:.1f}s"
            )

            if val_r2 > best_r2:
                best_r2 = val_r2
                torch.save(
                    {"state_dict": model.state_dict(), "r2": val_r2, "epoch": epoch},
                    best_path,
                )

        results.append({"fold": fold, "best_r2": best_r2})

    print("\nFold results:", results)

    return results


# =========================
# Инференс
# =========================
def load_checkpoints(checkpoint_dir: str) -> List[str]:
    ckpts = []

    for name in sorted(os.listdir(checkpoint_dir)):
        if name.startswith("fold_") and name.endswith(".pt"):
            ckpts.append(os.path.join(checkpoint_dir, name))
    if not ckpts:
        raise FileNotFoundError(f"Нет чекпойнтов вида fold_*.pt в {checkpoint_dir}")

    return ckpts


def run_inference(
    test_df: pd.DataFrame, state2idx: Dict[str, int], species2idx: Dict[str, int], args
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_paths = load_checkpoints(args.checkpoint_dir)

    model = PastureModel(
        num_states=max(len(state2idx), 1),
        num_species=max(len(species2idx), 1),
        num_numeric=len(test_df.iloc[0]["num_feats"]),
        backbone_name=args.backbone,
        pretrained=False,  # загрузим веса из чекпойнта
        hidden_dim=args.hidden_dim,
    ).to(device)

    base_tf = build_transforms(args.image_size, "val")
    tta_transforms = [base_tf]

    if args.tta > 1:
        tta_transforms.append(
            transforms.Compose(
                [
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        )

    # Заранее готовим sample_id порядок
    sample_ids = []

    for _, row in test_df.iterrows():
        image_id = row["image_id"]
        for t in TARGET_ORDER:
            sample_ids.append(f"{image_id}__{t}")

    preds_accum = np.zeros((len(test_df), len(TARGET_ORDER)), dtype=np.float64)

    for ckpt in ckpt_paths:
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        tta_preds = np.zeros_like(preds_accum)

        for tf in tta_transforms:
            ds = PastureDataset(test_df, args.data_dir, tf, is_train=False)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            all_preds = []
            with torch.no_grad():
                for batch in loader:
                    images = batch["image"].to(device)
                    state_idx = batch["state_idx"].to(device)
                    species_idx = batch["species_idx"].to(device)
                    num_feats = batch["num_feats"].to(device)
                    out = model(images, state_idx, species_idx, num_feats)
                    all_preds.append(out.cpu().numpy())
            fold_preds = np.concatenate(all_preds, axis=0)
            tta_preds += fold_preds
        tta_preds /= len(tta_transforms)
        preds_accum += tta_preds

    preds_accum /= len(ckpt_paths)

    # Разворачиваем в long-формат
    rows = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        image_id = row["image_id"]
        for j, t in enumerate(TARGET_ORDER):
            rows.append(
                {"sample_id": f"{image_id}__{t}", "target": float(preds_accum[i, j])}
            )
    submission = pd.DataFrame(rows)
    submission.to_csv(args.submission, index=False)
    print(f"Submission сохранен в {args.submission} ({len(submission)} строк)")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="CSIRO Image2Biomass baseline")
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Путь к каталогу с train.csv/test.csv и папками изображений",
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Куда сохранять чекпойнты (train)"
    )
    parser.add_argument(
        "--checkpoint-dir", default="outputs", help="Откуда брать чекпойнты (infer)"
    )
    parser.add_argument(
        "--submission",
        default="submission.csv",
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

    train_df, state2idx, species2idx, num_stats = load_train_dataframe(args.data_dir)
    if args.mode == "train":
        run_training(train_df, state2idx, species2idx, num_stats, args)
    else:
        test_df = load_test_dataframe(args.data_dir, state2idx, species2idx, num_stats)
        run_inference(test_df, state2idx, species2idx, args)


if __name__ == "__main__":
    main()
