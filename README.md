![logo](docs/assets/tf.net.logo.png)

**TensorFlow.NET** (TF.NET) provides a .NET Standard binding for [TensorFlow](https://www.tensorflow.org/). It aims to implement the complete Tensorflow API in C# which allows .NET developers to develop, train and deploy Machine Learning models with the cross-platform .NET Standard framework. TensorFlow.NET has built-in Keras high-level interface and is released as an independent package [TensorFlow.Keras](https://www.nuget.org/packages/TensorFlow.Keras/).

[![Discord](https://img.shields.io/discord/1106946823282761851?label=Discord)](https://discord.gg/qRVm82fKTS)
[![QQ群聊](https://img.shields.io/static/v1?label=QQ&message=群聊&color=brightgreen)](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=sN9VVMwbWjs5L0ATpizKKxOcZdEPMrp8&authKey=RLDw41bLTrEyEgZZi%2FzT4pYk%2BwmEFgFcrhs8ZbkiVY7a4JFckzJefaYNW6Lk4yPX&noverify=0&group_code=985366726)
[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sci-sharp/community)
[![CI Status](https://github.com/SciSharp/TensorFlow.NET/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/SciSharp/TensorFlow.NET/actions/workflows/build_and_test.yml)
[![Documentation Status](https://readthedocs.org/projects/tensorflownet/badge/?version=latest)](https://tensorflownet.readthedocs.io/en/latest/?badge=latest)
[![TensorFlow.NET Badge](https://img.shields.io/nuget/v/TensorFlow.NET?label=TensorFlow.NET)](https://www.nuget.org/packages/TensorFlow.NET)
[![TensorFlow.Keras Badge](https://img.shields.io/nuget/v/TensorFlow.Keras?label=TensorFlow.Keras)](https://www.nuget.org/packages/TensorFlow.Keras)
[![MyGet Badge](https://img.shields.io/badge/dynamic/json?color=purple&label=Nightly%20Release&prefix=myget-v&query=items%5B0%5D.lower&url=https%3A%2F%2Fwww.myget.org%2FF%2Fscisharp%2Fapi%2Fv3%2Fregistration1%2Ftensorflow.net%2Findex.json)](https://www.myget.org/feed/scisharp/package/nuget/Tensorflow.NET)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/en_US)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/javiercp/BinderTF.NET/master?urlpath=lab)

English | [中文](docs/README-CN.md)

> [!IMPORTANT]
> We're happy that our work on tensorflow.net has attracted many users. However, at this time, none of the main maintainers of this repo is available for new features and bug fix. We won't refuse PRs and will help to review them.
> 
> If you would like to be a contributor or maintainer of tensorflow.net, we'd like to help you to start up.
>
> We feel sorry for that and we'll resume the maintaining for this project once one of us has bandwidth for it.
>   

*master branch and v0.100.x is corresponding to tensorflow v2.10, v0.6x branch is from tensorflow v2.6, v0.15-tensorflow1.15 is from tensorflow1.15. Please add `https://www.myget.org/F/scisharp/api/v3/index.json` to nuget source to use nightly release.*


![tensors_flowing](docs/assets/tensors_flowing.gif)

## Why Tensorflow.NET ?

`SciSharp STACK`'s mission is to bring popular data science technology into the .NET world and to provide .NET developers with a powerful Machine Learning tool set without reinventing the wheel. Since the APIs are kept as similar as possible you can immediately adapt any existing TensorFlow code in C# or F# with a zero learning curve. Take a look at a comparison picture and see how comfortably a TensorFlow/Python script translates into a C# program with TensorFlow.NET.

![python vs csharp](docs/assets/syntax-comparision.png)

SciSharp's philosophy allows a large number of machine learning code written in Python to be quickly migrated to .NET, enabling .NET developers to use cutting edge machine learning models and access a vast number of TensorFlow resources which would not be possible without this project.

In comparison to other projects, like for instance [TensorFlowSharp](https://www.nuget.org/packages/TensorFlowSharp/) which only provide TensorFlow's low-level C++ API and can only run models that were built using Python, Tensorflow.NET makes it possible to build the pipeline of training and inference with pure C# and F#. Besides, Tensorflow.NET provides binding of Tensorflow.Keras to make it easy to transfer your code from python to .NET.

[ML.NET](https://github.com/dotnet/machinelearning) also take Tensorflow.NET as one of the backends to train and infer your model, which provides better integration with .NET.

## Documention

Introduction and simple examples：[Tensorflow.NET Documents](https://scisharp.github.io/tensorflow-net-docs)

Detailed documention：[The Definitive Guide to Tensorflow.NET](https://tensorflownet.readthedocs.io/en/latest/FrontCover.html)

Examples：[TensorFlow.NET Examples](https://github.com/SciSharp/TensorFlow.NET-Examples)

Troubleshooting of running example or installation：[Tensorflow.NET FAQ](tensorflowlib/README.md)

## Usage

### Installation

You can search the package name in NuGet Manager, or use the commands below in package manager console.

The installation contains two parts, the first is the main body:

```sh
### Install Tensorflow.NET
PM> Install-Package TensorFlow.NET

### Install Tensorflow.Keras
PM> Install-Package TensorFlow.Keras
```

The second part is the computing support part. Only one of the following packages is needed, depending on your device and system.

```
### CPU version for Windows and Linux
PM> Install-Package SciSharp.TensorFlow.Redist

### CPU version for MacOS
PM> Install-Package SciSharp.TensorFlow.Redist-OSX

### GPU version for Windows (CUDA and cuDNN are required)
PM> Install-Package SciSharp.TensorFlow.Redist-Windows-GPU

### GPU version for Linux (CUDA and cuDNN are required)
PM> Install-Package SciSharp.TensorFlow.Redist-Linux-GPU
```


Two simple examples are given here to introduce the basic usage of Tensorflow.NET. As you can see, it's easy to write C# code just like that in Python.

### Example - Linear Regression in `Eager` mode

```csharp
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.NumPy;

// Parameters        
var training_steps = 1000;
var learning_rate = 0.01f;
var display_step = 100;

// Sample data
var X = np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
             7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f);
var Y = np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
             2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f);
var n_samples = X.shape[0];

// We can set a fixed init value in order to demo
var W = tf.Variable(-0.06f, name: "weight");
var b = tf.Variable(-0.73f, name: "bias");
var optimizer = keras.optimizers.SGD(learning_rate);

// Run training for the given number of steps.
foreach (var step in range(1, training_steps + 1))
{
    // Run the optimization to update W and b values.
    // Wrap computation inside a GradientTape for automatic differentiation.
    using var g = tf.GradientTape();
    // Linear regression (Wx + b).
    var pred = W * X + b;
    // Mean square error.
    var loss = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples);
    // should stop recording
    // Compute gradients.
    var gradients = g.gradient(loss, (W, b));

    // Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, (W, b)));

    if (step % display_step == 0)
    {
        pred = W * X + b;
        loss = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples);
        print($"step: {step}, loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}");
    }
}
```

Run this example in [Jupyter Notebook](https://github.com/SciSharp/SciSharpCube).

### Example - Toy version of `ResNet` in `Keras` functional API

```csharp
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.NumPy;

var layers = keras.layers;
// input layer
var inputs = keras.Input(shape: (32, 32, 3), name: "img");
// convolutional layer
var x = layers.Conv2D(32, 3, activation: "relu").Apply(inputs);
x = layers.Conv2D(64, 3, activation: "relu").Apply(x);
var block_1_output = layers.MaxPooling2D(3).Apply(x);
x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_1_output);
x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
var block_2_output = layers.Add().Apply(new Tensors(x, block_1_output));
x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_2_output);
x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
var block_3_output = layers.Add().Apply(new Tensors(x, block_2_output));
x = layers.Conv2D(64, 3, activation: "relu").Apply(block_3_output);
x = layers.GlobalAveragePooling2D().Apply(x);
x = layers.Dense(256, activation: "relu").Apply(x);
x = layers.Dropout(0.5f).Apply(x);
// output layer
var outputs = layers.Dense(10).Apply(x);
// build keras model
var model = keras.Model(inputs, outputs, name: "toy_resnet");
model.summary();
// compile keras model in tensorflow static graph
model.compile(optimizer: keras.optimizers.RMSprop(1e-3f),
    loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
    metrics: new[] { "acc" });
// prepare dataset
var ((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data();
// normalize the input
x_train = x_train / 255.0f;
// training
model.fit(x_train[new Slice(0, 2000)], y_train[new Slice(0, 2000)],
            batch_size: 64,
            epochs: 10,
            validation_split: 0.2f);
// save the model
model.save("./toy_resnet_model");
```

The F# example for linear regression is available [here](docs/Example-fsharp.md).

More adcanced examples could be found in [TensorFlow.NET Examples](https://github.com/SciSharp/TensorFlow.NET-Examples).

## Version Relationships

| TensorFlow.NET Versions                 | tensorflow 1.14, cuda 10.0 | tensorflow 1.15, cuda 10.0 | tensorflow 2.3, cuda 10.1 | tensorflow 2.4, cuda 11 | tensorflow 2.7, cuda 11 |tensorflow 2.10, cuda 11 |
| -------------------------- | ------------- | -------------- | ------------- | ------------- | ------------ | ------------ |
| tf.net 0.10x, tf.keras 0.10 |  |  |  |  |  | x |
| tf.net 0.7x, tf.keras 0.7   |  |  |  |  | x |  |
| tf.net 0.4x, tf.keras 0.5   |  |  |  | x |  |  |
| tf.net 0.3x, tf.keras 0.4   |  |  | x |  |  |  |
| tf.net 0.2x                 |  | x | x |  |  |  |
| tf.net 0.15                 | x | x |  |  |  |  |
| tf.net 0.14                 | x |  |  |  |  |  |


```
tf.net 0.4x -> tf native 2.4 
tf.net 0.6x -> tf native 2.6      
tf.net 0.7x -> tf native 2.7
tf.net 0.10x -> tf native 2.10
...
```

## Contribution:

Feel like contributing to one of the hottest projects in the Machine Learning field? Want to know how Tensorflow magically creates the computational graph? 

We appreciate every contribution however small! There are tasks for novices to experts alike, if everyone tackles only a small task the sum of contributions will be huge.

You can:
- Star Tensorflow.NET or share it with others
- Tell us about the missing APIs compared to Tensorflow
- Port Tensorflow unit tests from Python to C# or F#
- Port Tensorflow examples to C# or F# and raise issues if you come accross missing parts of the API or BUG
- Debug one of the unit tests that is marked as Ignored to get it to work
- Debug one of the not yet working examples and get it to work
- Help us to complete the documentions.


#### How to debug unit tests:

The best way to find out why a unit test is failing is to single step it in C# or F# and its corresponding Python at the same time to see where the flow of execution digresses or where variables exhibit different values. Good Python IDEs like PyCharm let you single step into the tensorflow library code. 

#### Git Knowhow for Contributors

Add SciSharp/TensorFlow.NET as upstream to your local repo ...
```git
git remote add upstream git@github.com:SciSharp/TensorFlow.NET.git
```

Please make sure you keep your fork up to date by regularly pulling from upstream. 
```git
git pull upstream master
```

### Support
Buy our book to make open source project be sustainable [TensorFlow.NET实战](https://item.jd.com/13441549.html)
<p float="left">
<img src="https://user-images.githubusercontent.com/1705364/198852429-91741881-c196-401e-8e9e-2f8656196613.png" width="250" />
<img src="https://user-images.githubusercontent.com/1705364/198852521-2f842043-3ace-49d2-8533-039c6a043a3f.png" width="260" />
<img src="https://user-images.githubusercontent.com/1705364/198852721-54cd9e7e-9210-4931-a86c-77584b25b8e1.png" width="260" />
</p>

### Contact

Join our chat on [Discord](https://discord.gg/qRVm82fKTS) or [Gitter](https://gitter.im/sci-sharp/community).

Follow us on [Twitter](https://twitter.com/ScisharpStack), [Facebook](https://www.facebook.com/scisharp.stack.9), [Medium](https://medium.com/scisharp), [LinkedIn](https://www.linkedin.com/company/scisharp-stack/).

TensorFlow.NET is a part of [SciSharp STACK](https://scisharp.github.io/SciSharp/)
<br>
<a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp-stack.png" width="391" height="100" /></a>
