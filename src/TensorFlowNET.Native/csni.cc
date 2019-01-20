#include <windows.h>
#include <iostream>
#include "c_api.h"
typedef char* (__stdcall *TFFunc)();

int main() {
  HINSTANCE hinstLib = LoadLibrary(TEXT("libtensorflow.dll"));
  if (!hinstLib) {
    std::cout << "could not load the dynamic library" << std::endl;
    return EXIT_FAILURE;
  }

  TFFunc version = (TFFunc) GetProcAddress(hinstLib, "TF_Version");
  if (!version) {
    std::cout << "could not locate the function" << std::endl;
    return EXIT_FAILURE;
  }
  
  printf("Hello from TensorFlow C library version %s", version());
  return 0;
}