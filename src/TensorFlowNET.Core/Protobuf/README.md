### Download compiler from https://github.com/protocolbuffers/protobuf/releases
```shell
set SRC_DIR=D:\Projects\tensorflow
set DST_DIR=D:\Projects\TensorFlow.NET\src\TensorFlowNET.Core\Protobuf

protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\framework\resource_handle.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\framework\tensor_shape.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\framework\types.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\framework\tensor.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\framework\attr_value.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\framework\node_def.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\framework\versions.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\framework\function.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\framework\graph.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\protobuf\saver.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\core\protobuf\meta_graph.proto
protoc -I=%SRC_DIR% --csharp_out=%DST_DIR% tensorflow\python\training\checkpoint_state.proto
```
