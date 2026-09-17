# Native ONNX Runtime probe

`ort_native_benchmark.cc` is deliberately a narrow C++ probe. It measures the
native `Session::Run` boundary only, with the same float tensors, CPU provider,
thread count and conservative low-memory options as the Python reference. It
writes the raw C++ outputs so the Python driver can reject a result unless every
float is identical.

It is not a replacement for `app/detector` or `app/inpaint`: preprocessing,
detector decoding, mask authority and all safety policy remain in Python. A
faster model-call number alone must not be treated as a production migration.

Run it with an external ONNX Runtime 1.24.1 Linux x64 package and external
models (never commit either):

```bash
python scripts/benchmark_native_ort.py \
  --kind detector \
  --model /path/to/bubble_yolo.onnx \
  --image /path/to/representative.png \
  --ort-root /path/to/onnxruntime-linux-x64-1.24.1 \
  --out benchmark-results/native-ort-detector
```

The driver builds the C++ probe with `g++`, records median/min/P95 timings in
`report.json`, and exits non-zero if a native output differs from Python.
Results compare `CPUExecutionProvider` only; they do not compare or supersede
the production detector's OpenVINO provider path.
