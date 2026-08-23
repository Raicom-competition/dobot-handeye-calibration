import argparse
import random
import shutil
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def emit(message):
    print(message, flush=True)


def read_classes(class_file):
    classes = []
    for line in Path(class_file).read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            classes.append(name)
    return classes


def find_pairs(images_dir, labels_dir=None):
    images_dir = Path(images_dir)
    label_index = {}
    if labels_dir:
        for txt in Path(labels_dir).rglob("*.txt"):
            if txt.stat().st_size > 0:
                label_index[txt.stem.lower()] = txt

    pairs = []
    seen = set()
    for image in sorted(images_dir.rglob("*")):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue

        label = image.with_suffix(".txt")
        if not (label.exists() and label.stat().st_size > 0):
            label = label_index.get(image.stem.lower())

        if label is not None and label.exists() and label.stat().st_size > 0:
            key = str(image.resolve()).lower()
            if key not in seen:
                seen.add(key)
                pairs.append((image, label))
    return pairs


def collect_class_ids(pairs):
    ids = set()
    for _, label in pairs:
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                try:
                    ids.add(int(float(parts[0])))
                except ValueError:
                    pass
    return ids


def remap_label_line(line, remap):
    parts = line.split()
    if not parts:
        return line
    try:
        old = int(float(parts[0]))
        parts[0] = str(remap.get(old, old))
    except ValueError:
        pass
    return " ".join(parts)


def split_dataset(images_dir, labels_dir, out_dir, val_ratio=0.2, seed=42):
    out_dir = Path(out_dir)
    pairs = find_pairs(images_dir, labels_dir)
    if not pairs:
        return 0, 0, {}

    remap = {old: new for new, old in enumerate(sorted(collect_class_ids(pairs)))}

    random.Random(seed).shuffle(pairs)
    val_count = max(1, int(len(pairs) * val_ratio))
    groups = {"train": pairs[val_count:], "val": pairs[:val_count]}

    for split, group in groups.items():
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        for path in (image_dir, label_dir):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

        for image, label in group:
            stem = "%s_%s" % (image.parent.name, image.stem)
            stem = stem.replace(" ", "_")
            shutil.copy(image, image_dir / (stem + image.suffix))
            lines = [
                remap_label_line(line, remap)
                for line in label.read_text(encoding="utf-8").splitlines()
            ]
            (label_dir / (stem + ".txt")).write_text(
                "\n".join(lines), encoding="utf-8"
            )

    return len(groups["train"]), len(groups["val"]), remap


def write_yaml(out_dir, classes, yaml_path):
    import yaml

    data = {
        "path": str(Path(out_dir)).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    Path(yaml_path).write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def find_best_pt(project_dir):
    candidates = sorted(
        Path(project_dir).rglob("weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="handeye_train")
    parser.add_argument("--best", required=True)
    args = parser.parse_args()

    classes = read_classes(args.classes)
    if not classes:
        emit("[ERROR] 类别文件为空，请先填写 class.txt")
        sys.exit(1)
    emit("[CLASSES] %s" % ", ".join(classes))

    emit("[SPLIT] 正在扫描图片目录 %s" % args.images)
    if args.labels:
        emit("[SPLIT] 标签目录 %s" % args.labels)
    train_count, val_count, remap = split_dataset(args.images, args.labels, args.out)
    if train_count == 0:
        emit("[ERROR] 没有找到“图片 + 同名 .txt 标签”的有效样本")
        emit("[HINT] 图片放 raw/，标签放 raw/ 或 labels/，且文件名要一致")
        sys.exit(1)
    emit("[SPLIT] 训练集 %d 张 / 验证集 %d 张" % (train_count, val_count))
    if remap:
        emit("[REMAP] 标签类别重新编号: %s" % remap)
    if len(remap) > len(classes):
        emit(
            "[ERROR] 标签里出现 %d 个类别，但 class.txt 只声明了 %d 个，请补齐 class.txt"
            % (len(remap), len(classes))
        )
        sys.exit(1)
    if len(remap) < len(classes):
        emit(
            "[WARN] class.txt 声明了 %d 个类别，但标签里只用到 %d 个"
            % (len(classes), len(remap))
        )

    write_yaml(args.out, classes, args.yaml)
    emit("[YAML] %s" % args.yaml)

    emit(
        "[TRAIN] 开始训练 epochs=%d imgsz=%d batch=%d workers=%d device=%s"
        % (args.epochs, args.imgsz, args.batch, args.workers, args.device)
    )

    import torch
    from ultralytics import YOLO

    if args.device == "0" and not torch.cuda.is_available():
        emit("[ERROR] 已选择 GPU，但 torch 检测不到 CUDA，已安全终止。")
        sys.exit(1)
    device = args.device
    emit("[DEVICE] %s" % ("GPU (CUDA)" if device == "0" else "CPU"))

    model = YOLO(args.model)
    model.train(
        data=args.yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        amp=False,
    )

    best_src = find_best_pt(args.project)
    if best_src is None:
        emit("[ERROR] 未找到训练生成的 best.pt")
        sys.exit(1)

    shutil.copy(best_src, args.best)
    emit("[BEST] %s" % args.best)
    emit("[DONE] 训练完成")


if __name__ == "__main__":
    main()
