### How to compile CSharp Native Interface


git clone https://github.com/tensorflow/tensorflow

`cd tensorflow/tensorflow`

copy `csharp` folder to `tensorflow`, the csharp folder should be in the same parent directory with other language binding.

`cd csharp`

`bazel build //tensorflow/csharp:csni`