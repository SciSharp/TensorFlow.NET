# Comprehensive Roadmap: Next Version Release of TensorFlow.NET (TF 2.16 - 2.22+ Parity)

## Context & Objectives
This document establishes the official engineering roadmap to release the next major generation of **TensorFlow.NET** (`TensorFlow.NET`, `TensorFlow.Keras`, and `SciSharp.TensorFlow.Redist`). 

The overarching goal is to achieve architectural, mathematical, and behavioral parity with modern Python TensorFlow (v2.16 LTS through v2.22+ located in `refs/py-tensorflow-sot`), transitioning to a modern **.NET 8.0 / .NET 9.0** high-performance runtime baseline while preserving backward compatibility for existing enterprise consumers.

---

## Architectural Decisions & Foundations (Settled Design Tree)

1. **Target SOT Baseline**: Staged rollout starting with **TensorFlow 2.16 LTS** (Milestone 1: Keras 3 engine + C-API modernization) progressing to **TensorFlow 2.22** (Milestone 2: Sub-byte FP8/Int4 quantization + latest TSL kernels).
2. **Runtime Framework**: Pure **.NET 8.0 / .NET 9.0** baseline. Drops `.NET Standard 2.0` and `.NET 6.0` legacy constraints to leverage `System.Runtime.Intrinsics` (AVX-512 / ARM Neon), zero-copy `Span<T>` / `Memory<T>`, and native `Half` / `Int128`.
3. **Keras Modernization**: Complete clean-room rewrite of **Keras 3** as primary (`Tensorflow.Keras.*`), featuring pure symbolic `KerasTensor` DAGs, stateless layers, and universal `.keras` zip archive serialization. Legacy Keras 2.x code is segregated into `Tensorflow.Keras.Legacy` with `[Obsolete]` migration guides for seamless retro-compatibility.
4. **Gradient Parity**: Programmatic generation of all ~382 op gradients via a dedicated C# Roslyn CLI tool (`tools/TensorFlow.GradientGen`) parsing Python SOT AST.
5. **Memory & Buffer Lifecycle**: Zero-copy unmanaged memory managers (`UnmanagedMemoryManager<T>`), strict `SafeHandle` wrapping for native pointers, and SIMD hardware acceleration.
6. **Testing & Parity Verification**: Hybrid verification strategy using static golden fixtures (`.npy`/`.json`) for 100% headless CI runs and a live Python SOT dual-runner tool for active development.

---

## Phased Implementation Roadmap

### Phase 1: Runtime Baseline & Foundational C-API / DTypes [COMPLETED]
- [x] **1.1 Project & Solution Modernization (.NET 8.0 / 9.0)**: Target `<TargetFrameworks>net8.0;net9.0</TargetFrameworks>`, `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>`, C# 13.
- [x] **1.2 Native C-API Bindings & Status Payloads**: Implement `TF_SetPayload`, `TF_ForEachPayload`, `TF_SetStatusFromIOError`, `TF_TensorBitcastFrom`, `TF_TensorIsAligned`, `TF_TensorDefaultAlignment`, `TF_TensorElementCount`.
- [x] **1.3 DType Modernization & Bug Fixes**: Correct `float16`/`bfloat16` mappings in `dtypes.cs`, add full FP8 and sub-byte type enum definitions (`TF_FLOAT8_E5M2`..`TF_FLOAT4_E2M1FN`).

---

### Phase 2: Op Gradients Programmatic Generator & NumPy Parity [UPCOMING]

#### 2.1 Programmatic Gradient Generation Tool (`tools/TensorFlow.GradientGen`)
- Develop standalone CLI tool `tools/TensorFlow.GradientGen` to:
  - Parse Python AST and gradient registration decorators (`@ops.RegisterGradient`) across all files in `refs/py-tensorflow-sot/tensorflow/python/ops/*_grad.py`.
  - Emit idiomatic C# partial classes in `src/TensorFlowNET.Core/Gradients/Generated/` implementing `[RegisterGradient("OpName")]`.
  - Expand registration from current 92 ops to full SOT parity (~382 ops) across math, nn, image, linalg, sparse, and tensor arrays.
- **Second-Order Gradients**:
  - Implement nested gradient formulations for `SoftsignGrad`, `ReluGrad`, `TanhGrad`, `SoftplusGrad`, `SigmoidGrad`, `SqrtGrad`, and `FusedBatchNormGrad`.

#### 2.2 NumPy Operators & Multi-Axis Slicing Parity
- **Numerical Parity (`np.isclose` & `np.allclose`)**:
  - Implement complete tolerance math in `src/TensorFlowNET.Core/NumPy/Numpy.cs` and `NumPy.Logical.cs`:
    $$\text{diff} \le \text{atol} + \text{rtol} \times |b|$$
  - Implement integer overflow prevention (`maximum(a, b) - minimum(a, b)`) and `equal_nan` handling.
- **Dynamic Multidimensional Slicing (`src/TensorFlowNET.Core/Tensors/Tensor.Indexing.cs`)**:
  - Add full support for Ellipsis (`...`), `NewAxis` / `None`, negative step strides, and dynamic tensor-valued slice indices using the native `StridedSlice` kernel.

---

### Phase 3: Clean-Room Keras 3 Engine & Modern Serialization

#### 3.1 Symbolic Tracing DAG (`KerasTensor`)
- Decouple Keras functional construction from active native C-API `tf.Graph` instances.
- Introduce `KerasTensor` representing symbolic shapes, dtypes, and inbound/outbound node connections.
- Implement topological sort in `src/TensorFlowNET.Keras/Engine/Functional.cs` to resolve `Functional` execution graphs purely from `KerasTensor` outputs.
- Move legacy graph-coupled Keras 2 classes to `src/TensorFlowNET.Keras/Legacy/` (`Tensorflow.Keras.Legacy` namespace) and tag with `[Obsolete]` migration attributes.

#### 3.2 Universal `.keras` Zip Archive Serialization
- Implement native `.keras` zip archive reading and writing in `src/TensorFlowNET.Keras/Saving/`:
  - `config.json`: Model topology and layer hyperparameters.
  - `metadata.json`: Keras version, build timestamp, framework signatures.
  - `model.weights.h5` / `variables.safetensors`: Serialized weight arrays.
- Align `get_config()` and `from_config()` serialization schemas across all layers, optimizers, losses, and metrics with Python Keras 3.
- Maintain legacy loaders for SavedModel (`saved_model.pb`) and `.h5`.

#### 3.3 Modular Step Execution Loop
- Refactor `Model` execution pipeline into modular, overridable step methods:
  - `train_step(data)`
  - `test_step(data)`
  - `predict_step(data)`
- Refactor `LossesContainer` and `MetricsContainer` for consistent reduction, masking, and sample weight support.

---

### Phase 4: Verification, Parity Test Harness & Release Pipeline

#### 4.1 Hybrid Parity Test Harness
- **Static Golden Fixtures (`test/TensorFlowNET.UnitTest/Parity/`)**:
  - Generate golden test vectors (`.npy`/`.json`) from Python SOT for all newly generated op gradients, NumPy operators, and Keras 3 layers.
  - Ensure fast, deterministic, 100% headless CI test execution.
- **Live Dual-Runner Tool (`tools/TensorFlow.ParityRunner`)**:
  - CLI tool running side-by-side execution in C# and Python (via local `.venv`), asserting identical outputs within numerical tolerances for new ops.

#### 4.2 Release Packaging & Publishing
- Bump version to `v0.200.0` (or `v1.0.0`) in `Directory.Build.props`.
- Update GitHub Actions CI/CD workflows (`.github/workflows/build_and_test.yml`, `release.yml`):
  - Add multi-platform matrix builds (Windows, Linux, macOS ARM64/x64).
  - Package and publish nightly builds to MyGet and stable packages to NuGet (`TensorFlow.NET`, `TensorFlow.Keras`, `SciSharp.TensorFlow.Redist`).

---

### Key File Locations for Changes

| Subsystem | Target Files |
| :--- | :--- |
| **Project & Build** | `Directory.Build.props`, `TensorFlow.NET.sln`, `.github/workflows/build_and_test.yml` |
| **C-API & Status** | `src/TensorFlowNET.Core/Status/c_api.status.cs`, `Status.cs`, `c_api.tensor.cs` |
| **DTypes & Memory** | `src/TensorFlowNET.Core/Tensors/dtypes.cs`, `TF_DataType.cs`, `Tensor.cs` |
| **Gradient Generator** | `tools/TensorFlow.GradientGen/`, `src/TensorFlowNET.Core/Gradients/Generated/` |
| **NumPy & Slicing** | `src/TensorFlowNET.Core/NumPy/Numpy.cs`, `NumPy.Logical.cs`, `Tensor.Indexing.cs` |
| **Keras 3 Engine** | `src/TensorFlowNET.Keras/Engine/KerasTensor.cs`, `Functional.cs`, `Layer.cs`, `Model.cs` |
| **Keras Legacy** | `src/TensorFlowNET.Keras/Legacy/` |
| **Serialization** | `src/TensorFlowNET.Keras/Saving/KerasZipSaver.cs`, `Saving/KerasZipLoader.cs` |
| **Test Harness** | `tools/TensorFlow.ParityRunner/`, `test/TensorFlowNET.UnitTest/Parity/` |
