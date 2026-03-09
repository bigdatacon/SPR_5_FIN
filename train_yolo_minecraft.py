# train_yolo_minecraft.py
from ultralytics import YOLO
import os
import argparse
import yaml

def train_model(data_path, epochs=100, imgsz=640, batch=8, model_name='yolov8n.pt', experiment_name='rsp_baseline'):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден!")

    model = YOLO(model_name)

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=experiment_name,
        patience=20,
        seed=42,
        project="artifacts"  # <-- сюда сохраняются все результаты
    )
# run 
# python train_yolo_minecraft.py --data datasets/minecraft/data.yaml --epochs 12 --imgsz 512 --batch 4 --model yolov8s.pt --name yolo_minecraft_cuda
# python evaluate_yolo.py --model artifacts/yolo_minecraft_cpu/weights/best.pt --data datasets/minecraft/data.yaml --split test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели YOLOv8 на Minecraft')
    parser.add_argument('--data', type=str, default=r'datasets/minecraft/data.yaml', help='Путь к data.yaml')
    parser.add_argument('--epochs', type=int, default=1, help='Количество эпох обучения')
    parser.add_argument('--imgsz', type=int, default=512, help='Размер изображения')
    parser.add_argument('--batch', type=int, default=4, help='Размер батча')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Название предобученной модели')
    parser.add_argument('--name', type=str, default='yolo_minecraft_cpu', help='Название эксперимента')

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
        experiment_name=args.name
    )

