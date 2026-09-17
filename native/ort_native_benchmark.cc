// Minimal ONNX Runtime C++ benchmark used by scripts/benchmark_native_ort.py.
//
// Inputs are supplied in a TSV manifest to avoid adding a JSON dependency:
//   input_name<TAB>1,3,512,512<TAB>/absolute/path/input.f32
//
// The program intentionally measures only Session::Run.  It does not claim to
// replace image preprocessing, detector postprocessing, or the safe inpainting
// policy in the Python pipeline.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string model;
  std::string manifest;
  std::string output_dir;
  int warmup = 3;
  int runs = 7;
  int intra_op = 1;
};

struct Input {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

std::vector<std::string> Split(const std::string& value, char delimiter) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (start <= value.size()) {
    const size_t end = value.find(delimiter, start);
    fields.push_back(value.substr(start, end - start));
    if (end == std::string::npos) {
      return fields;
    }
    start = end + 1;
  }
  return fields;
}

int ParsePositiveInt(const std::string& text, const char* option) {
  try {
    const int value = std::stoi(text);
    if (value <= 0) {
      throw std::invalid_argument("not positive");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string(option) + " must be a positive integer");
  }
}

int ParseNonNegativeInt(const std::string& text, const char* option) {
  try {
    const int value = std::stoi(text);
    if (value < 0) {
      throw std::invalid_argument("negative");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string(option) + " must be a non-negative integer");
  }
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string option(argv[i]);
    if (option == "--model" || option == "--manifest" ||
        option == "--output-dir" || option == "--warmup" ||
        option == "--runs" || option == "--intra-op") {
      if (++i >= argc) {
        throw std::runtime_error("missing value for " + option);
      }
      const std::string value(argv[i]);
      if (option == "--model") args.model = value;
      if (option == "--manifest") args.manifest = value;
      if (option == "--output-dir") args.output_dir = value;
      if (option == "--warmup") args.warmup = ParseNonNegativeInt(value, "--warmup");
      if (option == "--runs") args.runs = ParsePositiveInt(value, "--runs");
      if (option == "--intra-op") args.intra_op = ParsePositiveInt(value, "--intra-op");
      continue;
    }
    if (option == "--help") {
      std::cout << "usage: ort_native_benchmark --model MODEL --manifest INPUTS.tsv "
                   "--output-dir DIR [--warmup 3] [--runs 7] [--intra-op 1]\\n";
      std::exit(0);
    }
    throw std::runtime_error("unknown option: " + option);
  }
  if (args.model.empty() || args.manifest.empty() || args.output_dir.empty()) {
    throw std::runtime_error("--model, --manifest and --output-dir are required");
  }
  return args;
}

size_t ElementCount(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    throw std::runtime_error("tensor shape cannot be empty");
  }
  size_t count = 1;
  for (const int64_t dimension : shape) {
    if (dimension <= 0) {
      throw std::runtime_error("all input dimensions must be positive");
    }
    const size_t converted = static_cast<size_t>(dimension);
    if (count > std::numeric_limits<size_t>::max() / converted) {
      throw std::runtime_error("tensor shape overflow");
    }
    count *= converted;
  }
  return count;
}

std::vector<Input> LoadInputs(const std::string& manifest_path) {
  std::ifstream manifest(manifest_path);
  if (!manifest) {
    throw std::runtime_error("cannot read input manifest: " + manifest_path);
  }

  std::vector<Input> inputs;
  std::string line;
  int line_number = 0;
  while (std::getline(manifest, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#') continue;
    const auto fields = Split(line, '\t');
    if (fields.size() != 3 || fields[0].empty() || fields[1].empty() || fields[2].empty()) {
      throw std::runtime_error("invalid input manifest line " + std::to_string(line_number));
    }
    Input input;
    input.name = fields[0];
    for (const std::string& dimension : Split(fields[1], ',')) {
      input.shape.push_back(static_cast<int64_t>(ParsePositiveInt(dimension.c_str(), "input dimension")));
    }
    const size_t expected_count = ElementCount(input.shape);
    std::ifstream binary(fields[2], std::ios::binary | std::ios::ate);
    if (!binary) {
      throw std::runtime_error("cannot read input tensor: " + fields[2]);
    }
    const std::streamsize bytes = binary.tellg();
    if (bytes != static_cast<std::streamsize>(expected_count * sizeof(float))) {
      throw std::runtime_error("wrong byte length for input tensor: " + fields[2]);
    }
    binary.seekg(0);
    input.data.resize(expected_count);
    if (!binary.read(reinterpret_cast<char*>(input.data.data()), bytes)) {
      throw std::runtime_error("failed reading input tensor: " + fields[2]);
    }
    inputs.push_back(std::move(input));
  }
  if (inputs.empty()) {
    throw std::runtime_error("input manifest contains no tensors");
  }
  return inputs;
}

double Percentile(std::vector<double> values, const double percentile) {
  if (values.empty()) throw std::runtime_error("no measurements");
  std::sort(values.begin(), values.end());
  const double position = percentile * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

void WriteOutput(const std::filesystem::path& output_path, const Ort::Value& value) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("only float outputs are supported by this benchmark");
  }
  const size_t count = info.GetElementCount();
  const float* data = value.GetTensorData<float>();
  std::ofstream binary(output_path, std::ios::binary);
  if (!binary) throw std::runtime_error("cannot write output: " + output_path.string());
  binary.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
  if (!binary) throw std::runtime_error("failed writing output: " + output_path.string());

  std::ofstream shape_file(output_path.string() + ".shape");
  if (!shape_file) throw std::runtime_error("cannot write output shape");
  const auto shape = info.GetShape();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) shape_file << ',';
    shape_file << shape[i];
  }
  shape_file << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const std::vector<Input> inputs = LoadInputs(args.manifest);
    std::filesystem::create_directories(args.output_dir);

    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "manga-translator-native-benchmark");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetIntraOpNumThreads(args.intra_op);
    options.SetInterOpNumThreads(1);
    // Match the conservative CPU session used for LaMa/fallback inference.
    options.DisableCpuMemArena();
    options.DisableMemPattern();
    Ort::Session session(environment, args.model.c_str(), options);

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t model_input_count = session.GetInputCount();
    if (model_input_count != inputs.size()) {
      throw std::runtime_error("input manifest count differs from model input count");
    }
    std::vector<std::string> input_names_storage;
    std::vector<const char*> input_names;
    input_names_storage.reserve(model_input_count);
    input_names.reserve(model_input_count);
    for (size_t index = 0; index < model_input_count; ++index) {
      auto name = session.GetInputNameAllocated(index, allocator);
      if (name.get() != inputs[index].name) {
        throw std::runtime_error("input manifest order/name differs from model at index " + std::to_string(index));
      }
      input_names_storage.emplace_back(name.get());
      input_names.push_back(input_names_storage.back().c_str());
    }

    std::vector<std::string> output_names_storage;
    std::vector<const char*> output_names;
    output_names_storage.reserve(session.GetOutputCount());
    output_names.reserve(session.GetOutputCount());
    for (size_t index = 0; index < session.GetOutputCount(); ++index) {
      auto name = session.GetOutputNameAllocated(index, allocator);
      output_names_storage.emplace_back(name.get());
      output_names.push_back(output_names_storage.back().c_str());
    }

    const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_values;
    input_values.reserve(inputs.size());
    for (const Input& input : inputs) {
      input_values.emplace_back(Ort::Value::CreateTensor<float>(
          memory_info, const_cast<float*>(input.data.data()), input.data.size(),
          input.shape.data(), input.shape.size()));
    }

    for (int iteration = 0; iteration < args.warmup; ++iteration) {
      (void)session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
                        input_values.size(), output_names.data(), output_names.size());
    }

    std::vector<double> durations_ms;
    durations_ms.reserve(args.runs);
    std::vector<Ort::Value> last_outputs;
    for (int iteration = 0; iteration < args.runs; ++iteration) {
      const auto started = std::chrono::steady_clock::now();
      last_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
                                 input_values.size(), output_names.data(), output_names.size());
      const auto finished = std::chrono::steady_clock::now();
      durations_ms.push_back(std::chrono::duration<double, std::milli>(finished - started).count());
    }

    for (size_t index = 0; index < last_outputs.size(); ++index) {
      WriteOutput(std::filesystem::path(args.output_dir) / ("cpp_output_" + std::to_string(index) + ".f32"),
                  last_outputs[index]);
    }
    const double min_ms = *std::min_element(durations_ms.begin(), durations_ms.end());
    std::cout << std::fixed << std::setprecision(4)
              << "NATIVE_ORT_RESULT {\"runs\":" << args.runs
              << ",\"warmup\":" << args.warmup
              << ",\"intra_op_threads\":" << args.intra_op
              << ",\"min_ms\":" << min_ms
              << ",\"median_ms\":" << Percentile(durations_ms, 0.5)
              << ",\"p95_ms\":" << Percentile(durations_ms, 0.95)
              << ",\"outputs\":" << last_outputs.size() << "}" << std::endl;
    return 0;
  } catch (const Ort::Exception& error) {
    std::cerr << "ONNX Runtime error: " << error.what() << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << std::endl;
  }
  return 2;
}
