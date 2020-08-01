using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public delegate Tensor Activation(Tensor x);

    public class Activations
    {
        /// <summary>
        /// Linear activation function (pass-through).
        /// </summary>
        public Activation Linear = x => x;
    }
}
