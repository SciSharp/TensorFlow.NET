# Description:
# CSharp Native Interface library intended for implementing the
# TensorFlow .NET Standard API using the TensorFlow C library.
package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary"
)



tf_cc_binary(
    name = "csni",
    srcs = ["csni.cc"],
    deps = [
      "//tensorflow/c:c_api",
      "//tensorflow/csharp/eager:cswrap_tfe_lib"],
)
