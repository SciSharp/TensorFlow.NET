using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class LeakyReLUArgs : LayerArgs
    {
        /// <summary>
        /// Negative slope coefficient.
        /// </summary>
        public float Alpha { get; set; } = 0.3f;
    }
}
