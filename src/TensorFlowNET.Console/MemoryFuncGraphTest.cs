using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Functions;
using static Tensorflow.Binding;

namespace Tensorflow
{
    class MemoryFuncGraphTest
    {
        public Action<int, int> ConcreteFunction
            => (epoch, iterate) =>
            {
                var func = new ConcreteFunction(Guid.NewGuid().ToString());
                func.Enter();
                var input = tf.placeholder(tf.float32);
                var output = permutation(input);
                func.ToGraph(input, output);
                func.Exit();
            };

        Tensor permutation(Tensor tensor)
        {
            TensorShape shape = (8, 64, 64, 3);
            var images = np.arange(shape.size).astype(np.float32).reshape(shape.dims);
            return tf.constant(images);
        }
    }
}
