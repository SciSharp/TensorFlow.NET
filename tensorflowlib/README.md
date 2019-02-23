Here are some pre-built TensorFlow binaries you can use for each platform:

- Linux
  - CPU-only: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz
  - GPU-enabled: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.12.0.tar.gz
- Mac: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.12.0.tar.gz
- Windows: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.12.0.zip

https://www.tensorflow.org/install/source_windows
pacman -S git patch unzip

1. Build static library

`bazel build --config=opt //tensorflow:libtensorflow.so`

2. Build pip package

`bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`

3. Generate pip installation file

`bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg`

4. Install from local wheel file.

`pip install C:/tmp/tensorflow_pkg/tensorflow-1.13.0-cp36-cp36m-win_amd64.whl`