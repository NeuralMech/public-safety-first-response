import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "CCTV_Public_Safety_Detection_Colab.ipynb"


def lines(text: str):
    return [line + "\n" for line in dedent(text).strip("\n").splitlines()]


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def build_notebook():
    return {
        "cells": [
            md_cell(
                """
                # CCTV Public Safety Detection

                This notebook builds a lightweight YOLO-based **risk-object detector**
                that can reuse existing CCTV infrastructure for early public-safety recognition.

                Current demo scope:
                - `fire_with_labels.zip` -> `fire`
                - `weapon_with_labels.zip` -> `knife`

                Important note:
                - This notebook detects **risk objects** such as `fire` and `knife`.
                - It is **not** a final behavior-understanding model for violence, riot, or stabbing events.
                - In the full project concept, these classes are only representative examples under limited public datasets.
                  The intended system assumes a broader and richer public-safety dataset.
                """
            ),
            code_cell(
                """
                !pip -q install ultralytics pyyaml lxml scikit-learn
                """
            ),
            md_cell(
                """
                ## 1. Prepare zip files

                Choose one of the following:

                1. Put the zip files in Google Drive and set `ZIP_ROOT` accordingly.
                2. Upload the zip files directly to Colab and use `ZIP_ROOT = Path("/content")`.
                """
            ),
            code_cell(
                """
                from google.colab import drive
                drive.mount("/content/drive")
                """
            ),
            code_cell(
                """
                from pathlib import Path

                # Example 1: if your files are in Google Drive
                # ZIP_ROOT = Path("/content/drive/MyDrive")
                #
                # Example 2: if you uploaded directly to Colab
                ZIP_ROOT = Path("/content")

                FIRE_ZIP = ZIP_ROOT / "fire_with_labels.zip"
                WEAPON_ZIP = ZIP_ROOT / "weapon_with_labels.zip"

                WORKDIR = Path("/content/public_safety_detection")
                RAW_DIR = WORKDIR / "raw"
                MERGED_DIR = WORKDIR / "merged_yolo"
                RUNS_DIR = WORKDIR / "runs"

                print("FIRE_ZIP   =", FIRE_ZIP)
                print("WEAPON_ZIP =", WEAPON_ZIP)
                """
            ),
            code_cell(
                """
                assert FIRE_ZIP.exists(), f"fire zip not found: {FIRE_ZIP}"
                assert WEAPON_ZIP.exists(), f"weapon zip not found: {WEAPON_ZIP}"
                print("zip files found.")
                """
            ),
            md_cell(
                """
                ## 2. Build a unified YOLO dataset

                This stage does the following:
                - keeps the fire dataset in YOLO format
                - converts the weapon dataset from Pascal VOC XML to YOLO txt
                - keeps only `knife`
                - merges everything into a 2-class YOLO dataset: `fire`, `knife`
                """
            ),
            code_cell(
                """
                import shutil
                import zipfile
                import random
                import yaml
                import xml.etree.ElementTree as ET
                from collections import Counter
                from sklearn.model_selection import train_test_split

                random.seed(42)

                CLASS_NAMES = ["fire", "knife"]
                FIRE_CLASS_ID = 0
                KNIFE_CLASS_ID = 1

                def reset_dir(path: Path):
                    if path.exists():
                        shutil.rmtree(path)
                    path.mkdir(parents=True, exist_ok=True)

                def unzip_to(zip_path: Path, out_dir: Path):
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(out_dir)

                def find_image_by_stem(image_dir: Path, stem: str):
                    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                        candidate = image_dir / f"{stem}{ext}"
                        if candidate.exists():
                            return candidate
                    return None

                def remap_yolo_label_file(src_label: Path, dst_label: Path, class_id_map: dict):
                    out_lines = []
                    for raw_line in src_label.read_text(encoding="utf-8").splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        src_cls = int(parts[0])
                        if src_cls not in class_id_map:
                            continue
                        parts[0] = str(class_id_map[src_cls])
                        out_lines.append(" ".join(parts))
                    dst_label.write_text("\\n".join(out_lines), encoding="utf-8")
                    return len(out_lines)

                def voc_object_to_yolo_line(obj, width, height, cls_id):
                    bbox = obj.find("bndbox")
                    xmin = float(bbox.findtext("xmin"))
                    ymin = float(bbox.findtext("ymin"))
                    xmax = float(bbox.findtext("xmax"))
                    ymax = float(bbox.findtext("ymax"))

                    x_center = ((xmin + xmax) / 2.0) / width
                    y_center = ((ymin + ymax) / 2.0) / height
                    box_w = (xmax - xmin) / width
                    box_h = (ymax - ymin) / height

                    return f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"

                def parse_weapon_xml(xml_path: Path, keep_classes=None):
                    if keep_classes is None:
                        keep_classes = {"knife": KNIFE_CLASS_ID}

                    root = ET.parse(xml_path).getroot()
                    width = int(root.findtext("size/width"))
                    height = int(root.findtext("size/height"))

                    label_lines = []
                    class_counter = Counter()

                    for obj in root.findall("object"):
                        cls_name = (obj.findtext("name") or "").strip().lower()
                        class_counter[cls_name] += 1
                        if cls_name not in keep_classes:
                            continue
                        label_lines.append(
                            voc_object_to_yolo_line(obj, width, height, keep_classes[cls_name])
                        )

                    return label_lines, class_counter

                def copy_fire_split(src_root: Path, src_split: str, dst_split: str):
                    src_images = src_root / src_split / "images"
                    src_labels = src_root / src_split / "labels"
                    dst_images = MERGED_DIR / "images" / dst_split
                    dst_labels = MERGED_DIR / "labels" / dst_split
                    dst_images.mkdir(parents=True, exist_ok=True)
                    dst_labels.mkdir(parents=True, exist_ok=True)

                    image_count = 0
                    label_count = 0

                    for src_image in sorted(src_images.iterdir()):
                        if src_image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                            continue
                        src_label = src_labels / f"{src_image.stem}.txt"
                        if not src_label.exists():
                            continue

                        dst_image = dst_images / f"fire__{src_image.name}"
                        dst_label = dst_labels / f"fire__{src_label.name}"
                        shutil.copy2(src_image, dst_image)
                        kept = remap_yolo_label_file(
                            src_label,
                            dst_label,
                            class_id_map={0: FIRE_CLASS_ID},
                        )
                        if kept == 0:
                            dst_image.unlink(missing_ok=True)
                            dst_label.unlink(missing_ok=True)
                            continue

                        image_count += 1
                        label_count += kept

                    return image_count, label_count

                def build_weapon_split(raw_weapon_root: Path, train_ratio=0.85):
                    image_dir = raw_weapon_root / "Sohas_weapon-Detection" / "images"
                    train_xml_dir = raw_weapon_root / "Sohas_weapon-Detection" / "annotations" / "xmls"
                    test_xml_dir = raw_weapon_root / "Sohas_weapon-Detection" / "annotations_test" / "xmls"

                    trainval_list = (
                        raw_weapon_root / "Sohas_weapon-Detection" / "annotations" / "trainval.txt"
                    ).read_text(encoding="utf-8").splitlines()
                    test_list = (
                        raw_weapon_root / "Sohas_weapon-Detection" / "annotations_test" / "test.txt"
                    ).read_text(encoding="utf-8").splitlines()

                    trainval_names = [x.strip() for x in trainval_list if x.strip()]
                    test_names = [x.strip() for x in test_list if x.strip()]

                    usable_trainval = []
                    skipped_trainval = 0
                    for stem in trainval_names:
                        xml_path = train_xml_dir / f"{stem}.xml"
                        if not xml_path.exists():
                            continue
                        label_lines, _ = parse_weapon_xml(xml_path)
                        if not label_lines:
                            skipped_trainval += 1
                            continue
                        usable_trainval.append(stem)

                    usable_test = []
                    skipped_test = 0
                    for stem in test_names:
                        xml_path = test_xml_dir / f"{stem}.xml"
                        if not xml_path.exists():
                            continue
                        label_lines, _ = parse_weapon_xml(xml_path)
                        if not label_lines:
                            skipped_test += 1
                            continue
                        usable_test.append(stem)

                    train_names, val_names = train_test_split(
                        usable_trainval,
                        test_size=max(1, int(len(usable_trainval) * (1 - train_ratio))),
                        random_state=42,
                    )

                    split_to_names = {
                        "train": train_names,
                        "val": val_names,
                        "test": usable_test,
                    }
                    split_to_xml_dir = {
                        "train": train_xml_dir,
                        "val": train_xml_dir,
                        "test": test_xml_dir,
                    }

                    summary = {}
                    for split_name, stems in split_to_names.items():
                        dst_images = MERGED_DIR / "images" / split_name
                        dst_labels = MERGED_DIR / "labels" / split_name
                        dst_images.mkdir(parents=True, exist_ok=True)
                        dst_labels.mkdir(parents=True, exist_ok=True)

                        image_count = 0
                        label_count = 0
                        for stem in stems:
                            xml_path = split_to_xml_dir[split_name] / f"{stem}.xml"
                            src_image = find_image_by_stem(image_dir, stem)
                            if src_image is None or not xml_path.exists():
                                continue

                            label_lines, _ = parse_weapon_xml(xml_path)
                            if not label_lines:
                                continue

                            dst_image = dst_images / f"knife__{src_image.name}"
                            dst_label = dst_labels / f"knife__{stem}.txt"
                            shutil.copy2(src_image, dst_image)
                            dst_label.write_text("\\n".join(label_lines), encoding="utf-8")

                            image_count += 1
                            label_count += len(label_lines)

                        summary[split_name] = {
                            "images": image_count,
                            "labels": label_count,
                        }

                    summary["skipped_trainval_nonknife"] = skipped_trainval
                    summary["skipped_test_nonknife"] = skipped_test
                    return summary

                reset_dir(WORKDIR)
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                MERGED_DIR.mkdir(parents=True, exist_ok=True)
                RUNS_DIR.mkdir(parents=True, exist_ok=True)

                print("Extracting datasets...")
                unzip_to(FIRE_ZIP, RAW_DIR / "fire")
                unzip_to(WEAPON_ZIP, RAW_DIR / "weapon")

                fire_root = RAW_DIR / "fire" / "Fire-Detection"
                weapon_root = RAW_DIR / "weapon"

                for split in ["train", "val", "test"]:
                    (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
                    (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

                fire_summary = {}
                fire_map = {"train": "train", "valid": "val", "test": "test"}
                for src_split, dst_split in fire_map.items():
                    imgs, labels = copy_fire_split(fire_root, src_split, dst_split)
                    fire_summary[dst_split] = {"images": imgs, "labels": labels}

                weapon_summary = build_weapon_split(weapon_root)

                data_yaml = {
                    "path": str(MERGED_DIR),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": CLASS_NAMES,
                    "nc": len(CLASS_NAMES),
                }
                data_yaml_path = MERGED_DIR / "data.yaml"
                data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

                print("\\n=== FIRE SUMMARY ===")
                print(fire_summary)
                print("\\n=== WEAPON SUMMARY ===")
                print(weapon_summary)
                print("\\n=== DATA YAML ===")
                print(data_yaml_path.read_text(encoding="utf-8"))
                """
            ),
            code_cell(
                """
                from collections import Counter

                def count_labels(label_dir: Path):
                    counts = Counter()
                    for txt_path in label_dir.glob("*.txt"):
                        for line in txt_path.read_text(encoding="utf-8").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            cls_id = int(line.split()[0])
                            counts[CLASS_NAMES[cls_id]] += 1
                    return counts

                for split in ["train", "val", "test"]:
                    image_count = len(list((MERGED_DIR / "images" / split).glob("*")))
                    label_counts = count_labels(MERGED_DIR / "labels" / split)
                    print(f"[{split}] images={image_count}, labels={dict(label_counts)}")
                """
            ),
            code_cell(
                """
                import matplotlib.pyplot as plt
                import cv2
                import random

                def yolo_to_xyxy(line, width, height):
                    cls_id, xc, yc, w, h = line.split()
                    cls_id = int(cls_id)
                    xc, yc, w, h = map(float, [xc, yc, w, h])
                    x1 = int((xc - w / 2) * width)
                    y1 = int((yc - h / 2) * height)
                    x2 = int((xc + w / 2) * width)
                    y2 = int((yc + h / 2) * height)
                    return cls_id, x1, y1, x2, y2

                sample_images = random.sample(list((MERGED_DIR / "images" / "train").glob("*")), 6)
                plt.figure(figsize=(16, 10))

                for i, image_path in enumerate(sample_images, start=1):
                    label_path = MERGED_DIR / "labels" / "train" / f"{image_path.stem}.txt"
                    img = cv2.imread(str(image_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img.shape[:2]

                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(line, w, h)
                        color = (0, 255, 0) if cls_id == FIRE_CLASS_ID else (255, 0, 0)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            img,
                            CLASS_NAMES[cls_id],
                            (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                    plt.subplot(2, 3, i)
                    plt.imshow(img)
                    plt.axis("off")
                    plt.title(image_path.name[:40])

                plt.tight_layout()
                plt.show()
                """
            ),
            md_cell(
                """
                ## 3. Train a lightweight YOLO model

                Baseline:
                - `yolov8n.pt`

                Recommended workflow:
                - first run a smoke test with `EPOCHS = 1 ~ 3`
                - then increase epochs after confirming that the pipeline is stable
                """
            ),
            code_cell(
                """
                from ultralytics import YOLO
                import torch
                from pathlib import Path

                MODEL_NAME = "yolov8n.pt"
                EPOCHS = 3
                IMG_SIZE = 640
                BATCH = 16
                DEVICE = 0 if torch.cuda.is_available() else "cpu"

                print("torch.cuda.is_available() =", torch.cuda.is_available())
                if torch.cuda.is_available():
                    print("GPU =", torch.cuda.get_device_name(0))
                else:
                    print("GPU not found. Using CPU.")
                print("DEVICE =", DEVICE)

                model = YOLO(MODEL_NAME)
                train_results = model.train(
                    data=str(MERGED_DIR / "data.yaml"),
                    epochs=EPOCHS,
                    imgsz=IMG_SIZE,
                    batch=BATCH,
                    device=DEVICE,
                    project=str(RUNS_DIR),
                    name="fire_knife_yolov8n",
                    pretrained=True,
                    plots=True,
                )

                print("save_dir =", train_results.save_dir)
                best_model_path = Path(train_results.save_dir) / "weights" / "best.pt"
                print("best_model_path =", best_model_path)
                """
            ),
            code_cell(
                """
                from ultralytics import YOLO
                from pathlib import Path

                if "train_results" in globals():
                    run_dir = Path(train_results.save_dir)
                else:
                    run_dirs = sorted(
                        [p for p in RUNS_DIR.iterdir() if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    assert run_dirs, "No training run directory was found. Run training first."
                    run_dir = run_dirs[0]

                best_candidates = [run_dir / "weights" / "best.pt"] + list(run_dir.rglob("best.pt"))
                best_candidates = [p for p in best_candidates if p.exists()]
                assert best_candidates, f"Could not find best.pt under: {run_dir}"

                best_model_path = best_candidates[0]
                print("run_dir =", run_dir)
                print("best_model_path =", best_model_path)

                best_model = YOLO(str(best_model_path))
                metrics = best_model.val(data=str(MERGED_DIR / "data.yaml"), split="test")
                print(metrics)
                """
            ),
            code_cell(
                """
                import random

                test_image_dir = MERGED_DIR / "images" / "test"
                test_samples = random.sample(list(test_image_dir.glob("*")), 8)

                pred_dir = RUNS_DIR / "predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)

                prediction_results = best_model.predict(
                    source=[str(p) for p in test_samples],
                    conf=0.25,
                    save=True,
                    project=str(RUNS_DIR),
                    name="sample_predictions",
                    exist_ok=True,
                )

                print("Saved sample predictions to:", RUNS_DIR / "sample_predictions")
                """
            ),
            md_cell(
                """
                ## 4. Temporal verification and human-in-the-loop dispatch recommendation

                The detector output should not directly trigger a drone launch.
                A safer CCTV-side structure is:

                **detection -> temporal verification -> event scoring -> dispatch recommendation -> human final approval**
                """
            ),
            code_cell(
                """
                from collections import deque
                from dataclasses import dataclass, asdict
                from typing import List, Dict


                class TemporalEventGate:
                    def __init__(self, window_size=10, fire_hits=4, knife_hits=3):
                        self.window_size = window_size
                        self.fire_hits = fire_hits
                        self.knife_hits = knife_hits
                        self.fire_history = deque(maxlen=window_size)
                        self.knife_history = deque(maxlen=window_size)

                    def update(self, detections):
                        fire_found = any(det["class_name"] == "fire" for det in detections)
                        knife_found = any(det["class_name"] == "knife" for det in detections)

                        self.fire_history.append(int(fire_found))
                        self.knife_history.append(int(knife_found))

                        return {
                            "fire_alarm": sum(self.fire_history) >= self.fire_hits,
                            "knife_alarm": sum(self.knife_history) >= self.knife_hits,
                            "fire_count": sum(self.fire_history),
                            "knife_count": sum(self.knife_history),
                        }


                @dataclass
                class DispatchRecommendation:
                    level: int
                    label: str
                    score: float
                    dispatch_recommended: bool
                    human_approval_required: bool
                    summary: str
                    evidence: List[str]


                class HumanInLoopDispatchEngine:
                    LEVEL_NAMES = {
                        0: "관찰",
                        1: "확인",
                        2: "출동 권고",
                        3: "긴급 출동 권고",
                    }

                    def __init__(
                        self,
                        review_threshold=0.30,
                        dispatch_threshold=0.60,
                        urgent_threshold=0.85,
                        report_bonus=0.15,
                        hotspot_bonus=0.10,
                        uncertainty_penalty=0.10,
                    ):
                        self.review_threshold = review_threshold
                        self.dispatch_threshold = dispatch_threshold
                        self.urgent_threshold = urgent_threshold
                        self.report_bonus = report_bonus
                        self.hotspot_bonus = hotspot_bonus
                        self.uncertainty_penalty = uncertainty_penalty

                    @staticmethod
                    def _max_conf(detections: List[Dict], class_name: str) -> float:
                        vals = [
                            float(det.get("confidence", 0.0))
                            for det in detections
                            if det.get("class_name") == class_name
                        ]
                        return max(vals) if vals else 0.0

                    @staticmethod
                    def yolo_result_to_detections(result) -> List[Dict]:
                        detections = []
                        names = result.names
                        for box in result.boxes:
                            cls_id = int(box.cls.item())
                            detections.append({
                                "class_name": names[cls_id],
                                "confidence": float(box.conf.item()),
                            })
                        return detections

                    def evaluate(
                        self,
                        detections: List[Dict],
                        temporal_state: Dict,
                        external_report: bool = False,
                        operator_marked_hotspot: bool = False,
                    ) -> DispatchRecommendation:
                        fire_conf = self._max_conf(detections, "fire")
                        knife_conf = self._max_conf(detections, "knife")
                        person_conf = self._max_conf(detections, "person")

                        score = 0.0
                        evidence = []

                        if temporal_state.get("fire_alarm", False):
                            score = max(score, 0.55 + 0.35 * fire_conf)
                            evidence.append("화재 징후 누적")
                        elif fire_conf >= 0.25:
                            score = max(score, 0.15 + 0.35 * fire_conf)
                            evidence.append("화재 단발 탐지")

                        if temporal_state.get("knife_alarm", False):
                            score = max(score, 0.45 + 0.40 * knife_conf)
                            evidence.append("흉기 징후 누적")
                        elif knife_conf >= 0.20:
                            score = max(score, 0.10 + 0.35 * knife_conf)
                            evidence.append("흉기 단발 탐지")

                        if person_conf > 0.0 and knife_conf > 0.0:
                            score += 0.05
                            evidence.append("사람과 흉기 동시 탐지")

                        if external_report:
                            score += self.report_bonus
                            evidence.append("외부 신고 확인")

                        if operator_marked_hotspot:
                            score += self.hotspot_bonus
                            evidence.append("관제자 지정 주의 구역")

                        if detections and max(fire_conf, knife_conf) < 0.35 and not external_report:
                            score -= self.uncertainty_penalty

                        score = max(0.0, min(1.0, score))

                        if score >= self.urgent_threshold:
                            level = 3
                        elif score >= self.dispatch_threshold:
                            level = 2
                        elif score >= self.review_threshold:
                            level = 1
                        else:
                            level = 0

                        summary = "유의미한 위험 신호 없음" if not evidence else " + ".join(evidence[:2])

                        return DispatchRecommendation(
                            level=level,
                            label=self.LEVEL_NAMES[level],
                            score=score,
                            dispatch_recommended=(level >= 2),
                            human_approval_required=True,
                            summary=summary,
                            evidence=evidence,
                        )


                engine = HumanInLoopDispatchEngine()

                demo_scenarios = {
                    "knife_scenario": {
                        "stream": [
                            [{"class_name": "knife", "confidence": 0.41}],
                            [],
                            [{"class_name": "knife", "confidence": 0.58}],
                            [{"class_name": "knife", "confidence": 0.73}],
                            [],
                            [{"class_name": "knife", "confidence": 0.77}],
                        ],
                        "external_report_steps": {4, 5, 6},
                        "hotspot_steps": set(),
                    },
                    "fire_scenario": {
                        "stream": [
                            [{"class_name": "fire", "confidence": 0.31}],
                            [],
                            [{"class_name": "fire", "confidence": 0.52}],
                            [{"class_name": "fire", "confidence": 0.67}],
                            [{"class_name": "fire", "confidence": 0.82}],
                            [{"class_name": "fire", "confidence": 0.88}],
                        ],
                        "external_report_steps": {6},
                        "hotspot_steps": set(),
                    },
                }

                for scenario_name, config in demo_scenarios.items():
                    print(f"=== {scenario_name} ===")
                    gate = TemporalEventGate(window_size=10, fire_hits=4, knife_hits=3)

                    for step, detections in enumerate(config["stream"], start=1):
                        temporal_state = gate.update(detections)
                        recommendation = engine.evaluate(
                            detections,
                            temporal_state,
                            external_report=(step in config["external_report_steps"]),
                            operator_marked_hotspot=(step in config["hotspot_steps"]),
                        )

                        print(f"[{step}] {recommendation.label}")
                        print(f"  요약: {recommendation.summary}")
                        print(f"  점수: {recommendation.score:.2f}")
                        print(f"  출동 권고: {recommendation.dispatch_recommended}")
                    print()
                """
            ),
            code_cell(
                """
                # Optional: export the trained detector
                exported = best_model.export(format="onnx")
                print("exported:", exported)
                """
            ),
            md_cell(
                """
                ## Next steps

                Current 1st-stage outputs:
                - a lightweight CCTV detector for `fire` and `knife`
                - a human-in-the-loop recommendation layer built on
                  `detection -> temporal verification -> event scoring -> dispatch recommendation -> human approval`

                Recommended next extensions:
                1. add a `person` class
                2. add tracking (for example, ByteTrack)
                3. upgrade the event score using `knife + person proximity + abrupt motion`
                4. show snapshot, recent clip, evidence, and recommended drone/ETA in the operator UI
                5. connect the recommendation output to the drone dispatch approval workflow
                """
            ),
        ],
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
