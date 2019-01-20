# Description:
# CSharp Native Interface library intended for implementing the
# TensorFlow .NET Standard API using the TensorFlow C library.

licenses(["notice"])  # Apache 2.0

cc_import(
  name = "libtensorflow",
  hdrs = ["c_api.h"],
  interface_library = "libtensorflow.lib",
  shared_library = "libtensorflow.dll",
)

cc_binary(
    name = "csni",
    srcs = ["csni.cc"],
    deps = [":libtensorflow"],
    linkstatic = 0
)
