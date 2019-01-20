#include <iostream>
#include "tensorflow/c/c_api.h"

int main() {
  printf("Hello from TensorFlow C library version %s", TF_Version());
  return 0;
}