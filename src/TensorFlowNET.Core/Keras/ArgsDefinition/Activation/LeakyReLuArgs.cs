using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class LeakyReLuArgs : LayerArgs
    {
        /// <summary>
        /// Negative slope coefficient.
        /// </summary>
        public float Alpha { get; set; } = 0.3f;
    }
}
