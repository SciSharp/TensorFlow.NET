using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class DropoutArgs : LayerArgs
    {
        /// <summary>
        /// Float between 0 and 1. Fraction of the input units to drop.
        /// </summary>
        public float Rate { get; set; }

        /// <summary>
        /// 1D integer tensor representing the shape of the
        /// binary dropout mask that will be multiplied with the input.
        /// </summary>
        public TensorShape NoiseShape { get; set; }

        /// <summary>
        /// random seed.
        /// </summary>
        public int? Seed { get; set; }

        public bool SupportsMasking { get; set; }
    }
}
