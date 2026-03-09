from ultralytics import YOLO
import argparse
import os
import time
import pandas as pd
import glob
import yaml  # для чтения data.yaml

def evaluate_model(model_path, data_path, split='test', imgsz=512, batch=4, device='cpu'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель {model_path} не найдена!")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден!")

    # Загружаем модель
    model = YOLO(model_path)

    # Читаем data.yaml через PyYAML
    with open(data_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    print(f"[DEBUG] data.yaml content: {data_yaml}")

    if split not in data_yaml:
        raise FileNotFoundError(f"Split '{split}' не найден в data.yaml")
    
    # images_folder = os.path.join(line.split(':')[1].strip(), "images")
    # images_folder = "C:/Users/Admin/NEIRO/SPR_5_FIN/mmdetection/datasets/minecraft/test/images"
    images_folder = "datasets/minecraft/test/images"
    print(f"[DEBUG] Используем папку для split '{split}': {images_folder}")

    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Папка с изображениями {split} не найдена по пути: {images_folder}")

    # Список изображений
    img_paths = glob.glob(os.path.join(images_folder, "*.jpg"))
    print(f"[DEBUG] Найдено {len(img_paths)} изображений в {images_folder}")

    # Подсчет FPS
    start_time = time.time()
    for img in img_paths:
        _ = model(img, imgsz=imgsz, device=device)
    fps = len(img_paths) / (time.time() - start_time)

    # Оценка метрик
    metrics = model.val(
        data=data_path,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False
    )

    print("\n=== Результаты оценки ===")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Средняя точность: {metrics.box.p.mean():.4f}")
    print(f"Средняя полнота: {metrics.box.r.mean():.4f}")
    print(f"FPS: {fps:.2f}")

    # Сохраняем результаты
    df = pd.DataFrame({
        "mAP": [metrics.box.map],
        "mAP50": [metrics.box.map50],
        "Precision": [metrics.box.p.mean()],
        "Recall": [metrics.box.r.mean()],
        "FPS": [fps]
    })
    os.makedirs("artifacts/metrics", exist_ok=True)
    csv_path = os.path.join("artifacts/metrics", f"yolo_metrics_{split}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DEBUG] Метрики сохранены в {csv_path}")

# python evaluate_yolo.py --model artifacts/yolo_minecraft_cuda5/weights/best.pt --data datasets/minecraft/data.yaml --split test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Оценка модели YOLOv8 для детекции')
    parser.add_argument('--model', type=str, default='runs/detect/rsp_baseline/weights/best.pt', 
                        help='Путь к обученной модели')
    parser.add_argument('--data', type=str, default='datasets/minecraft/data.yaml', help='Путь к data.yaml')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'], 
                        help='Набор данных для оценки')
    parser.add_argument('--imgsz', type=int, default=512, help='Размер изображений')
    parser.add_argument('--batch', type=int, default=4, help='Размер батча')
    parser.add_argument('--device', type=str, default='cpu', help='Устройство: cpu или cuda')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )