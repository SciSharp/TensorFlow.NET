using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Multiply inputs by `scale` and adds `offset`.
    /// </summary>
    public class Rescaling : Layer
    {
        RescalingArgs args;
        Tensor scale;
        Tensor offset;

        public Rescaling(RescalingArgs args) : base(args)
        {
            this.args = args;
        }

        protected override Tensor call(Tensor inputs, bool is_training = false, Tensor state = null)
        {
            scale = math_ops.cast(args.Scale, args.DType);
            throw new NotImplementedException("");
        }
    }
}
