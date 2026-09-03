# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

### Prerequisites
- Target framework: `.NET 6.0` / `.NET Standard 2.0`
- Solution file: `TensorFlow.NET.sln`

### Build Commands
```powershell
# Restore dependencies
dotnet restore TensorFlow.NET.sln

# Build entire solution (Debug / Release)
dotnet build TensorFlow.NET.sln -c Debug
dotnet build TensorFlow.NET.sln -c Release
```

### Testing Commands
```powershell
# Run all unit test suites
dotnet test TensorFlow.NET.sln --verbosity normal

# Run a specific test project
dotnet test test/TensorFlowNET.UnitTest/Tensorflow.Binding.UnitTest.csproj
dotnet test test/TensorFlowNET.Keras.UnitTest/Tensorflow.Keras.UnitTest.csproj
dotnet test test/TensorFlowNET.Graph.UnitTest/TensorFlowNET.Graph.UnitTest.csproj
dotnet test test/TensorFlow.Kernel.UnitTest/TensorFlow.Kernel.UnitTest.csproj
dotnet test test/TensorFlowNET.Native.UnitTest/Tensorflow.Native.UnitTest.csproj

# Run a specific test by fully qualified name
dotnet test --filter "FullyQualifiedName=TensorFlowNET.UnitTest.Basics.EagerModeTest"

# Run tests matching a name pattern
dotnet test --filter "Name~EagerMode"
```

### Switching CPU / GPU Redist Packages for Local Testing
The tests depend on `tools/Tensorflow.UnitTest.RedistHolder` to provide the native TensorFlow C-API runtime binaries:
```powershell
# Default / CPU redist
dotnet add tools/Tensorflow.UnitTest.RedistHolder package SciSharp.TensorFlow.Redist

# Windows GPU redist (CUDA & cuDNN required)
dotnet remove tools/Tensorflow.UnitTest.RedistHolder package SciSharp.TensorFlow.Redist
dotnet add tools/Tensorflow.UnitTest.RedistHolder package SciSharp.TensorFlow.Redist-Windows-GPU
```

---

## Architecture & Code Structure

**TensorFlow.NET (TF.NET)** is a pure .NET Standard / C# binding and implementation of TensorFlow (targeting TensorFlow v2.10 C-API / Python parity).

### High-Level Subsystems

```
src/
├── TensorFlowNET.Core/          # Core binding layer (Tensorflow.Binding)
│   ├── CApi/                    # Low-level P/Invoke native interop (c_api.cs, tf_status, etc.)
│   ├── Tensors/                 # Tensor, TensorShape, DType, memory management & disposables
│   ├── Eager/                   # Eager execution engine, EagerTensor, Context
│   ├── Grafts / Gradients/      # Automatic differentiation (GradientTape)
│   ├── Operations/              # Math, array, nn, control flow graph & eager ops
│   ├── Variables/               # ResourceVariable, RefVariable, VariableScope
│   ├── NumPy/                   # NumPy array & tensor interoperability (np.* helpers)
│   └── Protobuf/                # Generated TensorFlow protobuf definitions (GraphDef, NodeDef, etc.)
│
├── TensorFlowNET.Keras/         # High-level Keras neural network API (Tensorflow.Keras)
│   ├── Engine/                  # Model, Layer, Sequential, Functional, Training/Eval loop
│   ├── Layers/                  # Core, Convolutional, Recurrent, Normalization, Reshaping layers
│   ├── Optimizers/              # Adam, SGD, RMSprop, AdaGrad and base optimizer
│   ├── Losses/ & Metrics/       # Loss functions (MSE, CrossEntropy) & evaluation metrics
│   ├── Initializers/            # Weight initializers (Glorot, He, RandomNormal, etc.)
│   └── Utils/                   # Keras serialization, dataset pipelines, shape inference
│
├── TensorFlowNET.Text/          # NLP text processing, tokenization, and embeddings
├── TensorflowNET.Hub/           # Pretrained TensorFlow Hub model loading and execution
└── TensorFlowNET.Recommenders/  # Recommender systems and ranking architectures
```

### Key Architectural Concepts
- **Source of Truth (SOT)**: The official Python TensorFlow codebase located at `refs/py-tensorflow-sot` (`D:\Users\samuel\source\repos\NT\refs\py-tensorflow-sot`) serves as the strict reference implementation for all mathematical behavior, operator semantics, graph/eager execution semantics, and API signatures. Reference files under `refs/` are read-only.
- **Branching, Tagging & Python TensorFlow Version Mapping**:
  - **Version Mapping Scheme**: TF.NET versions directly mirror TensorFlow Python versions:
    - `v0.150.x` / `v0.15.x` -> Python TensorFlow `v2.15` / `v1.15`
    - `master` / `v0.110.x` / `v0.100.x` -> Python TensorFlow `v2.11` / `v2.10`
    - `v0.6x` -> Python TensorFlow `v2.6`
    - `v0.40` -> Python TensorFlow `v2.4`
  - **Branches**:
    - `master`: Main development line targeting parity with TensorFlow 2.10+.
    - Maintenance / version branches (e.g. `v0.6x`, `v0.15-tensorflow1.15`): Target specific legacy TensorFlow releases.
  - **Tags**: Semantic versioning prefixed with `v` (e.g. `v0.110.4-Transformer-Model`, `v0.100.5`), used by CI/CD packaging to determine the base version.
  - **Releases & CI/CD Pipeline**:
    - Automatic packaging triggered on PR merge to `master` with `auto-release` label (`.github/workflows/release_prepare.yml` -> `release.yml`).
    - Produces `-nightly` packages pushed to MyGet feed (`https://www.myget.org/F/scisharp/api/v3/index.json`) and stable releases pushed to NuGet (`TensorFlow.NET`, `TensorFlow.Keras`, `Tensorflow.Hub`).
- **Python / C# Idiom Parity**: Standard static imports provide syntax matching Python TensorFlow:
  ```csharp
  using static Tensorflow.Binding;
  using static Tensorflow.KerasApi;
  using Tensorflow;
  using Tensorflow.NumPy;
  ```
- **Execution Modes**: Supports both `Eager` mode (imperative tensor operations) and `Graph` mode (`tf.Graph`, `tf.Session`, `tf.placeholder`).
- **Native Interop Lifecycle**: Native `c_api` allocations (`SafeHandle`, `TF_Tensor`, `TF_Status`, `TF_Operation`) are wrapped via `DisposableObject` and memory lifetime management in `TensorFlowNET.Core`.

---

## Roadmap & SOT Feature Gap Objectives

The official engineering roadmap targeting modern Python TensorFlow (v2.16 LTS through v2.22+) parity and .NET 8/9 runtime modernization is maintained in **[ROADMAP.md](ROADMAP.md)**.


