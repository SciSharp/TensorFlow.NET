# 	Get started with TensorFlow.NET

I would describe TensorFlow as an open source machine learning framework developed by Google which can be used to build neural networks and perform a variety of machine learning tasks. it works on data flow graph where nodes are the mathematical operations and the edges are the data in the form of tensor, hence the name Tensor-Flow. 



Let's run a classic HelloWorld program first and see if TensorFlow is running on .NET. I can't think of a simpler way to be a HelloWorld.



### Install the TensorFlow.NET SDK

TensorFlow.NET uses the .NET Standard 2.0 standard, so your new project Target Framework can be .NET Framework or .NET Core.  All the examples in this book are using .NET Core 2.2 and Microsoft Visual Studio Community 2017. To start building TensorFlow program you just need to download and install the .NET SDK (Software Development Kit). You have to download the latest .NET Core SDK from offical website: https://dotnet.microsoft.com/download.



1. New a project

   ![New Project](_static/new-project.png)

2. Choose Console App (.NET Core)

   ![Console App](_static/new-project-console.png)



```cmd
### install tensorflow C# binding
PM> Install-Package TensorFlow.NET

### Install tensorflow binary
### For CPU version - Using this for tutorials recommended
PM> Install-Package SciSharp.TensorFlow.Redist

### For GPU version (CUDA and cuDNN are required)
PM> Install-Package SciSharp.TensorFlow.Redist-Windows-GPU
```

### Start coding Hello World

After installing the TensorFlow.NET package, you can use the `using Tensorflow` to introduce the TensorFlow library.



```csharp
using System;
using Tensorflow;
using static Tensorflow.Binding;
namespace Tutorial
{
    class Program
    {
        static void Main(string[] args)
        {
            new HelloWorld().Run();
        }
    }

    /// <summary>
    /// Simple hello world using TensorFlow
    /// </summary>
    public class HelloWorld
    {
        public void Run()
        {
            /* Create a Constant op
               The op is added as a node to the default graph.

               The value returned by the constructor represents the output
               of the Constant op. */
            var hello = tf.constant("Hello, TensorFlow!");

            // Start tf session
            using (var sess = tf.Session())
            {
                // Run the op
                var result = sess.run(hello);
                Console.WriteLine((string)result);
            }
        }
    }
}
```
After CTRL + F5 run, you will get the output.
```cmd
2020-04-24 10:50:30.498955: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Hello, TensorFlow!
Press any key to continue . . .
```



