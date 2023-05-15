TensorFlow.NET pack all required libraries in architecture-specific assemblies folders per NuGet standard.

```powershell
PM> Install-Package TensorFlow.NET
PM> Install-Package SciSharp.TensorFlow.Redist
```

Add `<RuntimeIdentifier>win-x64</RuntimeIdentifier>` to a `PropertyGroup` in your `.csproj` when targeting `.NET 472`.

### Run in Linux

Download Linux pre-built library and unzip `libtensorflow.so` and `libtensorflow_framework.so` into current running directory.

To run image recognition in Linux, please ensure some prerequisite libraries is install.

```shell
sudo apt install libc6-dev 
sudo apt install libgdiplus
```

More information about [System.Drawing on Linux](<https://www.hanselman.com/blog/HowDoYouUseSystemDrawingInNETCore.aspx>).

### Run TensorFlow with GPU
Before running verify you installed  CUDA and cuDNN (TensorFlow v1.15 is compatible with CUDA v10.0 and cuDNN v7.4 , TensorFlow v2.x is compatible with CUDA v10.2 and cuDNN v7.65), and make sure the corresponding cuda version is compatible. 

#### Mac OS
There is no GPU support for macOS, in the future TensorFlow will support [Apple M1 chip](https://github.com/apple/tensorflow_macos).

#### GPU for Windows

```powershell
PM> Install-Package SciSharp.TensorFlow.Redist-Windows-GPU
```

#### GPU for Linux
```powershell
PM> Install-Package SciSharp.TensorFlow.Redist-Linux-GPU
```

Since NuGet limits file size for 250M, we can't ship Linux GPU version as NuGet, you can download the library from [Google TensorFlow Storage](https://storage.googleapis.com/tensorflow).

### Download prebuild binary manually

TensorFlow packages are built nightly and uploaded to GCS for all supported platforms. They are uploaded to the [libtensorflow-nightly](https://www.tensorflow.org/install/lang_c) GCS bucket and are indexed by operating system and date built.


### Build from source for Windows

https://www.tensorflow.org/install/source_windows

Download [Bazel 2.0.0](https://github.com/bazelbuild/bazel/releases/tag/2.0.0) to build tensorflow2.x. We build customized binary to export c_api from this [fork](https://github.com/SciSharp/tensorflow).

Set ENV `BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC`.

`pacman -S git patch unzip`

1. Build static library

`bazel build --output_base=C:/tmp/tfcompilation --config=opt //tensorflow:tensorflow`

2. Build pip package

`bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`

3. Generate pip installation file

`bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg`

4. Install from local wheel file.

`pip install C:/tmp/tensorflow_pkg/tensorflow-1.15.0-cp36-cp36m-win_amd64.whl`

### Build from source for MacOS

```shell
$ cd /usr/local/lib/bazel/bin
$ curl -LO https://release.bazel.build/3.7.2/release/bazel-3.7.2-darwin-x86_64
$ chmod +x bazel-3.7.2-darwin-x86_64
$ cd ~/Projects/tensorflow
$ bazel build --config=opt //tensorflow:tensorflow
```

### Build specific version for tf.net

https://github.com/SciSharp/tensorflow

For Linux version, these APIs symbols should also be put into `tensorflow/c/version_script.lds` to be exported. 
Please refer to commit `https://github.com/SciSharp/tensorflow/commit/58122da06be3e7707500ad889dfd5c760a3e0424`