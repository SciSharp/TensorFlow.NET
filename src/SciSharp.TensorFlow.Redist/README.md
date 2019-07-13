## SciSharp.TensorFlow.Redist ##


`SciSharp.TensorFlow.Redist` is  a migration from [Microsoft.ML.TensorFlow.Redist](https://github.com/dotnet/machinelearning/tree/release/1.2/src/Redist/Microsoft.ML.TensorFlow.Redist). [ML.NET](https://github.com/dotnet/machinelearning) team will not maintain the package since [ML.NET](https://www.nuget.org/packages/Microsoft.ML) v1.4.0 going forward.

```powershell
PM> Install-Package SciSharp.TensorFlow.Redist
```

https://www.nuget.org/packages/SciSharp.TensorFlow.Redist

Related merged [commits](https://github.com/SciSharp/TensorFlow.NET/commit/854a5ba61ad0e400623821236bd117cc24c6cb77).

####  Pack and Deploy ####

On Windows, the tar command does not support extracting archives with symlinks. So when `dotnet pack` runs on Windows it will only package the Windows binaries.

1. Run `dotnet pack` under `src/SciSharp.TensorFlow.Redist` directory in Linux.
2. Run `nuget push SciSharp.TensorFlow.Redist.1.14.0.nupkg -k APIKEY -s https://api.nuget.org/v3/index.json`



