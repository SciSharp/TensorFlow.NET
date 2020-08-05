using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public partial class Activations
    {
        /// <summary>
        /// Linear activation function (pass-through).
        /// </summary>
        public Activation Linear = (features, name) => features;
    }
}
