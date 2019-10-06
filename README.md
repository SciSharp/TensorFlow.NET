![logo](docs/assets/tf.net.logo.png)

**TensorFlow.NET** (TF.NET) provides a .NET Standard binding for [TensorFlow](https://www.tensorflow.org/). It aims to implement the complete Tensorflow API in C# which allows .NET developers to develop, train and deploy Machine Learning models with the cross-platform .NET Standard framework. 

[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sci-sharp/community)
[![Tensorflow.NET](https://ci.appveyor.com/api/projects/status/wx4td43v2d3f2xj6?svg=true)](https://ci.appveyor.com/project/Haiping-Chen/tensorflow-net)
[![codecov](https://codecov.io/gh/SciSharp/NumSharp/branch/master/graph/badge.svg)](https://codecov.io/gh/SciSharp/NumSharp)
[![NuGet](https://img.shields.io/nuget/dt/TensorFlow.NET.svg)](https://www.nuget.org/packages/TensorFlow.NET)
[![Documentation Status](https://readthedocs.org/projects/tensorflownet/badge/?version=latest)](https://tensorflownet.readthedocs.io/en/latest/?badge=latest)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/en_US)

TF.NET is a member project of [SciSharp STACK](https://github.com/SciSharp). <a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp_badge.png" width="200" height="200" align="right" /></a>


![tensors_flowing](docs/assets/tensors_flowing.gif)

### Why TensorFlow.NET ?

`SciSharp STACK`'s mission is to bring popular data science technology into the .NET world and to provide .NET developers with a powerful Machine Learning tool set without reinventing the wheel. Since the APIs are kept as similar as possible you can immediately adapt any existing Tensorflow code in C# with a zero learning curve. Take a look at a comparison picture and see how comfortably a   Tensorflow/Python script translates into a C# program with TensorFlow.NET.

![pythn vs csharp](docs/assets/syntax-comparision.png)

SciSharp's philosophy allows a large number of machine learning code written in Python to be quickly migrated to .NET, enabling .NET developers to use cutting edge machine learning models and access a vast number of Tensorflow resources which would not be possible without this project.

In comparison to other projects, like for instance TensorFlowSharp which only provide Tensorflow's low-level C++ API and can only run models that were built using Python, Tensorflow.NET also implements Tensorflow's high level API where all the magic happens. This computation graph building layer is still under active development. Once it is completely implemented you can build new Machine Learning models in C#. 

### How to use

Install TF.NET and TensorFlow binary through NuGet.
```sh
### install tensorflow C# binding
PM> Install-Package TensorFlow.NET

### Install tensorflow binary
### For CPU version
PM> Install-Package SciSharp.TensorFlow.Redist

### For GPU version (CUDA and cuDNN are required)
PM> Install-Package SciSharp.TensorFlow.Redist-Windows-GPU
```

Import TF.NET in your project.

```cs
using static Tensorflow.Binding;
```

Linear Regression:

```c#
// We can set a fixed init value in order to debug
var W = tf.Variable(-0.06f, name: "weight");
var b = tf.Variable(-0.73f, name: "bias");

// Construct a linear model
var pred = tf.add(tf.multiply(X, W), b);

// Mean squared error
var cost = tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * n_samples);

// Gradient descent
// Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

// Initialize the variables (i.e. assign their default value)
var init = tf.global_variables_initializer();

// Start training
using(tf.Session()) 
{
    // Run the initializer
    sess.run(init);

    // Fit all training data
    for (int epoch = 0; epoch < training_epochs; epoch++)
    {
        foreach (var (x, y) in zip<float>(train_X, train_Y))
            sess.run(optimizer, (X, x), (Y, y));

        // Display logs per epoch step
        if ((epoch + 1) % display_step == 0)
        {
            var c = sess.run(cost, (X, train_X), (Y, train_Y));
            Console.WriteLine($"Epoch: {epoch + 1} cost={c} " + $"W={sess.run(W)} b={sess.run(b)}");
        }
    }

    Console.WriteLine("Optimization Finished!");
    var training_cost = sess.run(cost, (X, train_X), (Y, train_Y));
    Console.WriteLine($"Training cost={training_cost} W={sess.run(W)} b={sess.run(b)}");

    // Testing example
    var test_X = np.array(6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f);
    var test_Y = np.array(1.84f, 2.273f, 3.2f, 2.831f, 2.92f, 3.24f, 1.35f, 1.03f);
    Console.WriteLine("Testing... (Mean square loss Comparison)");
    var testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * test_X.shape[0]),
                                (X, test_X), (Y, test_Y));
    Console.WriteLine($"Testing cost={testing_cost}");
    var diff = Math.Abs((float)training_cost - (float)testing_cost);
    Console.WriteLine($"Absolute mean square loss difference: {diff}");

    return diff < 0.01;
});
```

Run this example in [Jupyter Notebook](https://github.com/SciSharp/SciSharpCube).

Read the docs & book [The Definitive Guide to Tensorflow.NET](https://tensorflownet.readthedocs.io/en/latest/FrontCover.html).

There are many examples reside at [TensorFlow.NET Examples](https://github.com/SciSharp/TensorFlow.NET-Examples).

Troubleshooting of running example or installation, please  refer [here](tensorflowlib/README.md).

### Contribute:

Feel like contributing to one of the hottest projects in the Machine Learning field? Want to know how Tensorflow magically creates the computational graph? We appreciate every contribution however small. There are tasks for novices to experts alike, if everyone tackles only a small task the sum of contributions will be huge.

You can:
* Let everyone know about this project
* Port Tensorflow unit tests from Python to C#
* Port missing Tensorflow code from Python to C#
* Port Tensorflow examples to C# and raise issues if you come accross missing parts of the API
* Debug one of the unit tests that is marked as Ignored to get it to work
* Debug one of the not yet working examples and get it to work

### How to debug unit tests:

The best way to find out why a unit test is failing is to single step it in C# and its pendant Python at the same time to see where the flow of execution digresses or where variables exhibit different values. Good Python IDEs like PyCharm let you single step into the tensorflow library code. 

### Git Knowhow for Contributors

Add SciSharp/TensorFlow.NET as upstream to your local repo ...
```git
git remote add upstream git@github.com:SciSharp/TensorFlow.NET.git
```

Please make sure you keep your fork up to date by regularly pulling from upstream. 
```git
git pull upstream master
```

### Contact

Feel free to star or raise issue on [Github](https://github.com/SciSharp/TensorFlow.NET).

Follow us on [Medium](https://medium.com/scisharp).

Join our chat on [Gitter](https://gitter.im/sci-sharp/community).

Scan QR code to join Tencent TIM group:

![SciSharp STACK](docs/TIM.jpg)

TensorFlow.NET is a part of [SciSharp STACK](https://scisharp.github.io/SciSharp/)
<br>
<a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp-stack.png" width="391" height="100" /></a>
