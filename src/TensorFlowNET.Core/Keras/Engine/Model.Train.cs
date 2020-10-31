using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        Tensor step_function(OwnedIterator iterator)
        {
            var data = iterator.next();
            train_step(data[0], data[1]);
            throw new NotImplementedException("");
        }

        /// <summary>
        /// The logic for one training step.
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        Tensor train_step(Tensor x, Tensor y)
        {
            using var tape = tf.GradientTape();
            var y_pred = Apply(x, is_training: true);
            throw new NotImplementedException("");
        }
    }
}
