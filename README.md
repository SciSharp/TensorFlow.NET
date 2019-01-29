# TensorFlow.NET
TensorFlow.NET provides .NET Standard binding for [TensorFlow](https://www.tensorflow.org/).

[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sci-sharp/community)
[![Tensorflow.NET](https://ci.appveyor.com/api/projects/status/wx4td43v2d3f2xj6?svg=true)](https://ci.appveyor.com/project/Haiping-Chen/tensorflow-net)
[![codecov](https://codecov.io/gh/SciSharp/NumSharp/branch/master/graph/badge.svg)](https://codecov.io/gh/SciSharp/NumSharp)
[![NuGet](https://img.shields.io/nuget/dt/TensorFlow.NET.svg)](https://www.nuget.org/packages/TensorFlow.NET)
[![Documentation Status](https://readthedocs.org/projects/tensorflownet/badge/?version=latest)](https://tensorflownet.readthedocs.io/en/latest/?badge=latest)

TensorFlow.NET is a member project of [SciSharp](https://github.com/SciSharp) stack.

![tensors_flowing](docs/assets/tensors_flowing.gif)

### How to use
Install TensorFlow.NET through NuGet.
```sh
PM> Install-Package TensorFlow.NET
```

If you are using Linux or Mac OS, please download the pre-compiled dll [here](tensorflow.so) and place it in the working folder. This is only need for Linux and Mac OS, and already packed into NuGet for Windows.

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

Star me or raise issue on [Github](https://github.com/SciSharp/TensorFlow.NET) feel free.

Scan QR code to join TIM group:

![SciSharp STACK](docs/TIM.png)
