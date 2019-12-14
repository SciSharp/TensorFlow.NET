TensorFlow.NET pack all required libraries in architecture-specific assemblies folders per NuGet standard [Deprecated] .

We changed to use `Microsoft.ML.TensorFlow.Redist` to maintain the TensorFlow library.



### Download manually

Here are some pre-built TensorFlow binaries you can use for each platform:

- Linux
  - CPU-only: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz
  - GPU-enabled: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.14.0.tar.gz
- Mac: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.14.0.tar.gz
- Windows
  - CPU-only: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.14.0.zip
  - GPU-enabled: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-1.14.0.zip



### Run in Linux

`Install-Package TensorFlow.NET`

Download Linux pre-built library and unzip `libtensorflow.so` and `libtensorflow_framework.so` into current running directory.

To run image recognition in Linux, please ensure some prerequisite libraries is install.

```shell
sudo apt install libc6-dev 
sudo apt install libgdiplus
```

More information about [System.Drawing on Linux](<https://www.hanselman.com/blog/HowDoYouUseSystemDrawingInNETCore.aspx>).



### Run in Mac OS



### Tensorflow GPU for Windows

Before running verify you installed  CUDA and cuDNN (TensorFlow v1.14 is compatible with CUDA v10.0 and cuDNN v7.4), and make sure the corresponding cuda version is compatible. 

```powershell
PM> Install-Package SciSharp.TensorFlow.Redist-Windows-GPU
```



### Build from source for Windows

https://www.tensorflow.org/install/source_windows

`pacman -S git patch unzip`

1. Build static library

`bazel build --config=opt //tensorflow:libtensorflow.so`

2. Build pip package

`bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`

3. Generate pip installation file

`bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg`

4. Install from local wheel file.

`pip install C:/tmp/tensorflow_pkg/tensorflow-1.14.0-cp36-cp36m-win_amd64.whl`

### Export more APIs

Add more api to `c_api.h`

```c++
TF_CAPI_EXPORT extern void AddControlInput(TF_Graph* graph, TF_Operation* op, TF_Operation* input);
TF_CAPI_EXPORT extern void UpdateEdge(TF_Graph* graph, TF_Output new_src, TF_Input dst, TF_Status* status);
TF_CAPI_EXPORT extern void RemoveAllControlInputs(TF_Graph* graph, TF_Operation* op);
```

For Linux version, these APIs symbols should also be put into `tensorflow/c/version_script.lds` to be exported. 
Please refer to commit `https://github.com/SciSharp/tensorflow/commit/58122da06be3e7707500ad889dfd5c760a3e0424`
