from ultralytics import YOLO
import sys

def convert(pt_path: str, imgsz: int = 1024):
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=imgsz, opset=12)
    print("Xong. File .onnx nằm cùng thư mục với file .pt")

if __name__ == "__main__":
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "models/comic-speech-bubble-detector.pt"
    convert(pt_path)
