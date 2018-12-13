# TensorFlow.NET
TensorFlow.NET provides .NET Standard binding for [TensorFlow](https://www.tensorflow.org/).


TensorFlow.NET is a member project of SciSharp stack.

![tensors_flowing](docs/assets/tensors_flowing.gif)

### How to use
```cs
using tf = TensorFlowNET.Core.Tensorflow;

namespace TensorFlowNET.Examples
{
    public class HelloWorld : IExample
    {
        public void Run()
        {
            var hello = tf.constant("Hello, TensorFlow!");
        }
    }
}
```