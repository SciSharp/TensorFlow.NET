![logo](assets/tf.net.logo.png)

**Tensorflow.NET**是AI框架[TensorFlow](https://www.tensorflow.org/)在.NET平台上的实现，支持C#和F#，可以用来搭建深度学习模型并进行训练和推理，并内置了Numpy API，可以用来进行其它科学计算。

Tensorflow.NET并非对于Python的简单封装，而是基于C API的pure C#实现，因此使用时无需额外的环境，可以很方便地用NuGet直接安装使用。并且dotnet团队提供的[ML.NET](https://github.com/dotnet/machinelearning)也依赖于Tensorflow.NET，支持调用Tensorflow.NET进行训练和推理，可以很方便地融入.NET生态。

与tensorflow相同，Tensorflow.NET也内置了Keras这一高级API，只要在安装Tensorflow.NET的同时安装Tensorflow.Keras就可以使用，Keras支持以模块化的方式调用模型，给模型的搭建提供了极大的便利。

[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sci-sharp/community)
[![Tensorflow.NET](https://ci.appveyor.com/api/projects/status/wx4td43v2d3f2xj6?svg=true)](https://ci.appveyor.com/project/Haiping-Chen/tensorflow-net)
[![NuGet](https://img.shields.io/nuget/dt/TensorFlow.NET.svg)](https://www.nuget.org/packages/TensorFlow.NET)
[![Documentation Status](https://readthedocs.org/projects/tensorflownet/badge/?version=latest)](https://tensorflownet.readthedocs.io/en/latest/?badge=latest)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/en_US)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/javiercp/BinderTF.NET/master?urlpath=lab)

中文 | [English](https://github.com/SciSharp/TensorFlow.NET#readme)

*当前主分支与Tensorflow2.10版本相对应，支持Eager Mode，同时也支持v1的静态图。*


![tensors_flowing](assets/tensors_flowing.gif)

## Why Tensorflow.NET?

`SciSharp STACK`开源社区的目标是构建.NET平台下易用的科学计算库，而Tensorflow.NET就是其中最具代表性的仓库之一。在深度学习领域Python是主流，无论是初学者还是资深开发者，模型的搭建和训练都常常使用Python写就的AI框架，比如tensorflow。但在实际应用深度学习模型的时候，又可能希望用到.NET生态，亦或只是因为.NET是自己最熟悉的领域，这时候Tensorflow.NET就有显著的优点，因为它不仅可以和.NET生态很好地贴合，其API还使得开发者很容易将Python代码迁移过来。下面的对比就是很好的例子，Python代码和C#代码有着高度相似的API，这会使得迁移的时候无需做过多修改。

![python vs csharp](assets/syntax-comparision.png)

除了高度相似的API外，Tensorflow.NET与tensorflow也已经打通数据通道，tensorflow训练并保存的模型可以在Tensorflow.NET中直接读取并继续训练或推理，反之Tensorflow.NET保存的模型也可以在tensorflow中读取，这大大方便了模型的训练和部署。

与其它类似的库比如[TensorFlowSharp](https://www.nuget.org/packages/TensorFlowSharp/)相比，Tensorflow.NET的实现更加完全，提供了更多的高级API，使用起来更为方便，更新也更加迅速。


## 文档

基本介绍与简单用例：[Tensorflow.NET Documents](https://scisharp.github.io/tensorflow-net-docs)

详细文档：[The Definitive Guide to Tensorflow.NET](https://tensorflownet.readthedocs.io/en/latest/FrontCover.html)

例程：[TensorFlow.NET Examples](https://github.com/SciSharp/TensorFlow.NET-Examples)

运行例程常见问题：[Tensorflow.NET FAQ](tensorflowlib/README.md)

## 安装与使用

安装可以在NuGet包管理器中搜索包名安装，也可以用下面命令行的方式。

安装分为两个部分，第一部分是Tensorflow.NET的主体：

```sh
### 安装Tensorflow.NET
PM> Install-Package TensorFlow.NET

### 安装Tensorflow.Keras
PM> Install-Package TensorFlow.Keras
```

第二部分是计算支持部分，只需要根据自己的设备和系统选择下面之一即可：

```
### CPU版本，支持Windows、Linux和Mac
PM> Install-Package SciSharp.TensorFlow.Redist

### Windows下的GPU版本（需要安装CUDA和cuDNN）
PM> Install-Package SciSharp.TensorFlow.Redist-Windows-GPU

### Linux下的GPU版本（需要安装CUDA和cuDNN）
PM> Install-Package SciSharp.TensorFlow.Redist-Linux-GPU
```

下面给出两个简单的例子，更多例子可以在[TensorFlow.NET Examples]中查看。

### 简单例子（使用Eager Mode进行线性回归）

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

这一用例也可以在[Jupyter Notebook Example](https://github.com/SciSharp/SciSharpCube)进行运行.

### 简单例子（使用Keras搭建Resnet）

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

此外，Tensorflow.NET也支持用F#搭建上述模型进行训练和推理。

## Tensorflow.NET版本对应关系

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

如果使用过程中发现有缺失的版本，请告知我们，谢谢！

请注意Tensorflow.NET与Tensorflow.Keras版本存在一一对应关系，请安装与Tensorflow.NET对应的Tensorflow.Keras版本。

## 参与我们的开发:

我们欢迎任何人的任何形式的贡献！无论是文档中的错误纠正，新特性提议，还是BUG修复等等，都会使得Tensorflow.NET项目越来越好，Tensorflow.NET的全体开发者也会积极帮助解决您提出的问题。

下面任何一种形式都可以帮助Tensorflow.NET越来越好：

* Star和分享Tensorflow.NET项目
* 为Tensorflow.NET添加更多的用例
* 在issue中告知我们Tensorflow.NET目前相比tensorflow缺少的API或者没有对齐的特性
* 在issue中提出Tensorflow.NET存在的BUG或者可以改进的地方
* 在待办事项清单中选择一个进行或者解决某个issue
* 帮助我们完善文档，这也十分重要


## 支持我们
我们推出了[TensorFlow.NET实战](https://item.jd.com/13441549.html)这本书，包含了Tensorflow.NET主要开发者编写的讲解与实战例程，欢迎您的购买，希望这本书可以给您带来帮助。
<p float="left">
<img src="https://user-images.githubusercontent.com/1705364/198852429-91741881-c196-401e-8e9e-2f8656196613.png" width="250" />
<img src="https://user-images.githubusercontent.com/1705364/198852521-2f842043-3ace-49d2-8533-039c6a043a3f.png" width="260" />
<img src="https://user-images.githubusercontent.com/1705364/198852721-54cd9e7e-9210-4931-a86c-77584b25b8e1.png" width="260" />
</p>

## 联系我们

可以在 [Twitter](https://twitter.com/ScisharpStack), [Facebook](https://www.facebook.com/scisharp.stack.9), [Medium](https://medium.com/scisharp), [LinkedIn](https://www.linkedin.com/company/scisharp-stack/)中关注我们，也可以在[Gitter](https://gitter.im/sci-sharp/community)中与项目开发者以及其它使用者进行沟通交流，也欢迎在仓库中提起issue。

TensorFlow.NET is a part of [SciSharp STACK](https://scisharp.github.io/SciSharp/)
<br>
<a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp-stack.png" width="391" height="100" /></a>
