using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class DenseArgs : LayerArgs
    {
        /// <summary>
        /// Positive integer, dimensionality of the output space.
        /// </summary>
        public int Units { get; set; }

        /// <summary>
        /// Activation function to use.
        /// </summary>
        public Activation Activation { get; set; }

        /// <summary>
        /// Whether the layer uses a bias vector.
        /// </summary>
        public bool UseBias { get; set; } = true;

        /// <summary>
        /// Initializer for the `kernel` weights matrix.
        /// </summary>
        public IInitializer KernelInitializer { get; set; } = tf.glorot_uniform_initializer;

        /// <summary>
        /// Initializer for the bias vector.
        /// </summary>
        public IInitializer BiasInitializer { get; set; } = tf.zeros_initializer;

        /// <summary>
        /// Regularizer function applied to the `kernel` weights matrix.
        /// </summary>
        public IRegularizer KernelRegularizer { get; set; }

        /// <summary>
        /// Regularizer function applied to the bias vector.
        /// </summary>
        public IRegularizer BiasRegularizer { get; set; }

        /// <summary>
        /// Constraint function applied to the `kernel` weights matrix.
        /// </summary>
        public Action KernelConstraint { get; set; }

        /// <summary>
        /// Constraint function applied to the bias vector.
        /// </summary>
        public Action BiasConstraint { get; set; }
    }
}
