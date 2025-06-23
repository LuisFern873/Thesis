import os
import torch
import pickle
import json
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image

from torchvision import transforms
from tqdm import tqdm

CURRENT_DIR = Path(__file__).parent.absolute()
RAW_DIR = CURRENT_DIR / "raw"
TRAIN_DIR = RAW_DIR / "training"
TEST_DIR = RAW_DIR / "testing"
OUTPUT_DIR = CURRENT_DIR

if __name__ == "__main__":
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise RuntimeError("El dataset debe estar en `raw/training/` y `raw/testing/`.")

    image_size = int(input("Tamaño de imagen (por defecto 64): ") or 64)
    ratio = float(input("Proporción de datos por clase (por defecto 1.0): ") or 1.0)
    seed = int(input("Semilla aleatoria (por defecto 42): ") or 42)
    random.seed(seed)

    def collect_images(base_path):
        data = []
        classes = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        for cls_name in classes:
            class_dir = base_path / cls_name
            images = list(class_dir.glob("*.jpg"))
            random.shuffle(images)
            keep = int(len(images) * ratio)
            data += [(img_path.relative_to(CURRENT_DIR).as_posix(), class_to_idx[cls_name])
                     for img_path in images[:keep]]
        return data, classes

    print("Procesando datos de entrenamiento...")
    train_data, classes = collect_images(TRAIN_DIR)
    print("Procesando datos de prueba...")
    test_data, _ = collect_images(TEST_DIR)

    # Guardar etiquetas
    targets = {
        "train": [label for _, label in train_data],
        "test": [label for _, label in test_data],
    }
    torch_targets = torch.tensor(targets["train"] + targets["test"])
    torch.save(torch_targets, OUTPUT_DIR / "targets.pt")

    # Guardar lista de archivos
    filename_list = [path for path, _ in train_data + test_data]
    with open(OUTPUT_DIR / "filename_list.pkl", "wb") as f:
        pickle.dump(filename_list, f)

    # Guardar metadatos
    metadata = {
        "classes": classes,
        "image_size": image_size,
        "train_len": len(train_data),
        "test_len": len(test_data),
        "seed": seed,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Preprocesamiento completado.")
