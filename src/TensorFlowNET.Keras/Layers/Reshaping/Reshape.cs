using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
using System.Collections.Generic;
using System;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Layer that reshapes inputs into the given shape.
    /// </summary>
    public class Reshape : Layer
    {
        ReshapeArgs args;
        public Reshape(ReshapeArgs args)
            : base(args)
        {
            this.args = args;
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            var shape = new List<int> { inputs.shape[0] };
            shape.AddRange(args.TargetShape.dims);

            var result = array_ops.reshape(inputs, shape.ToArray());
            if (!tf.Context.executing_eagerly())
                // result = result.set_shape(compute_output_shape(inputs.shape));
                throw new NotImplementedException("");
            return result;
        }
    }
}
