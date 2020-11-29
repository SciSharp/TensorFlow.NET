using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Layers
{
    public partial class LayersApi
    {
        /// <summary>
        /// Layer that concatenates a list of inputs.
        /// </summary>
        /// <param name="axis">Axis along which to concatenate.</param>
        /// <returns></returns>
        public Concatenate Concatenate(int axis = -1)
            => new Concatenate(new MergeArgs
            {
                Axis = axis
            });
    }
}
