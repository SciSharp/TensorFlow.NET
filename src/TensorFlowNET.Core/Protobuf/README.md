```console
PM> Install-Package Google.Protobuf
```

#### How to generate `proto` files.

Download compiler from https://github.com/protocolbuffers/protobuf/releases.

Download `any.proto` from https://github.com/protocolbuffers/protobuf/tree/master/src/google/protobuf, place it at `google/protobuf/any.proto`.

Run `Gen.bat` under `src\TensorFlowNET.Core\Protobuf` folder.

