load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

cc_library(
    name = "cswrap_tfe_lib",
    srcs = [
        "cswrap_tfe_src.cc",
    ],
    hdrs = [
        "cswrap_tfe.h",
    ],
    visibility = [
        "//learning/deepmind/courier:__subpackages__",
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c:c_api",

    ],
)