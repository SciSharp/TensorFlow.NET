# TensorFlow.NET
TensorFlow.NET provides a .NET Standard binding for [TensorFlow](https://www.tensorflow.org/). It aims to implement the complete Tensorflow API in CSharp which allows .NET developers to develop, train and deploy Machine Learning models with the cross-platform .NET Standard framework. 

[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sci-sharp/community)
[![Tensorflow.NET](https://ci.appveyor.com/api/projects/status/wx4td43v2d3f2xj6?svg=true)](https://ci.appveyor.com/project/Haiping-Chen/tensorflow-net)
[![codecov](https://codecov.io/gh/SciSharp/NumSharp/branch/master/graph/badge.svg)](https://codecov.io/gh/SciSharp/NumSharp)
[![NuGet](https://img.shields.io/nuget/dt/TensorFlow.NET.svg)](https://www.nuget.org/packages/TensorFlow.NET)
[![Documentation Status](https://readthedocs.org/projects/tensorflownet/badge/?version=latest)](https://tensorflownet.readthedocs.io/en/latest/?badge=latest)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/en_US)

TensorFlow.NET is a member project of [SciSharp STACK](https://github.com/SciSharp).

![tensors_flowing](docs/assets/tensors_flowing.gif)

### Why should you use TensorFlow.NET ?

`SciSharp STASK`'s mission is to bring popular data science technology into the .NET world and to provide .NET developers with a powerful Machine Learning tool set without reinventing the wheel. Scince the APIs are kept as similar as possible you can immediately adapt any existing Tensorflow code in C# with a zero learning curve. Take a look at a comparison picture and see how comfortably a   Tensorflow/Python script translates into a C# program with TensorFlow.NET.

![pythn vs csharp](docs/assets/syntax-comparision.png)

SciSharp's philosophy allows a large number of machine learning code written in Python to be quickly migrated to .NET, enabling .NET developers to use cutting edge machine learning models and access a vast number of Tensorflow resources which would not be possible without this project.

In comparison to other projects, like for instance TensorFlowSharp which only provide Tensorflow's low-level C++ API and can only run models that were built using Python, Tensorflow.NET also implements Tensorflow's high level API where all the magic happens. This computation graph building layer is still under active development. Once it is completely implemented you can build new Machine Learning models in C#. 

### How to use

Install TensorFlow.NET through NuGet.
```sh
PM> Install-Package TensorFlow.NET
```

If you are using Linux or Mac OS, please download the pre-compiled dll [here](tensorflowlib) and place it in the working folder. This is only need for Linux and Mac OS, and already packed into NuGet for Windows.

Import tensorflow.net.

```cs
using Tensorflow;
```

Add two constants.
```cs
// Create a Constant op
var a = tf.constant(4.0f);
var b = tf.constant(5.0f);
var c = tf.add(a, b);

using (var sess = tf.Session())
{
    var o = sess.run(c);
}
```

Feed placeholder.
```cs
// Create a placeholder op
var a = tf.placeholder(tf.float32);
var b = tf.placeholder(tf.float32);
var c = tf.add(a, b);

using(var sess = tf.Session())
{
    var feed_dict = new Dictionary<Tensor, object>();
    feed_dict.Add(a, 3.0f);
    feed_dict.Add(b, 2.0f);

    var o = sess.run(c, feed_dict);
}
```

Read the docs & book [The Definitive Guide to Tensorflow.NET](https://tensorflownet.readthedocs.io/en/latest/FrontCover.html).

### More examples:

* [Hello World](test/TensorFlowNET.Examples/HelloWorld.cs)
* [Basic Operations](test/TensorFlowNET.Examples/BasicOperations.cs)
* [Image Recognition](test/TensorFlowNET.Examples/ImageRecognition.cs)
* [Linear Regression](test/TensorFlowNET.Examples/LinearRegression.cs)
* [Logistic Regression](test/TensorFlowNET.Examples/LogisticRegression.cs)
* [Nearest Neighbor](test/TensorFlowNET.Examples/NearestNeighbor.cs)
* [Text Classification](test/TensorFlowNET.Examples/TextClassificationWithMovieReviews.cs)
* [CNN Text Classification](test/TensorFlowNET.Examples/CnnTextClassification.cs)
* [Naive Bayes Classification](test/TensorFlowNET.Examples/NaiveBayesClassifier.cs)
* [Named Entity Recognition](test/TensorFlowNET.Examples/NamedEntityRecognition.cs)

Feel free to star or raise issue on [Github](https://github.com/SciSharp/TensorFlow.NET).

Scan QR code to join Tencent TIM group:

![SciSharp STACK](docs/TIM.jpg)

Or join our Chat on [Gitter](https://gitter.im/sci-sharp/community)
