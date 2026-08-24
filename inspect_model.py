import sys
import onnxruntime as ort


def main(path):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    print(f"model: {path}\n")

    print("--- inputs ---")
    for inp in session.get_inputs():
        print(f"name={inp.name} shape={inp.shape} type={inp.type}")

    print("\n--- outputs ---")
    for out in session.get_outputs():
        print(f"name={out.name} shape={out.shape} type={out.type}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/text_segmenter.onnx")
