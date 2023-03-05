using Newtonsoft.Json;
using System;
using System.Xml.Linq;
using Tensorflow.Operations.Initializers;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition
{
    // TODO: `activity_regularizer`
    public class DenseArgs : LayerArgs
    {
        /// <summary>
        /// Positive integer, dimensionality of the output space.
        /// </summary>
        [JsonProperty("units")]
        public int Units { get; set; }

        /// <summary>
        /// Activation function to use.
        /// </summary>
        [JsonProperty("activation")]
        public Activation Activation { get; set; }

        /// <summary>
        /// Whether the layer uses a bias vector.
        /// </summary>
        [JsonProperty("use_bias")]
        public bool UseBias { get; set; } = true;

        /// <summary>
        /// Initializer for the `kernel` weights matrix.
        /// </summary>
        [JsonProperty("kernel_initializer")]
        public IInitializer KernelInitializer { get; set; } = tf.glorot_uniform_initializer;

        /// <summary>
        /// Initializer for the bias vector.
        /// </summary>
        [JsonProperty("bias_initializer")]
        public IInitializer BiasInitializer { get; set; } = tf.zeros_initializer;

        /// <summary>
        /// Regularizer function applied to the `kernel` weights matrix.
        /// </summary>
        [JsonProperty("kernel_regularizer")]
        public IRegularizer KernelRegularizer { get; set; }

        /// <summary>
        /// Regularizer function applied to the bias vector.
        /// </summary>
        [JsonProperty("bias_regularizer")]
        public IRegularizer BiasRegularizer { get; set; }

        /// <summary>
        /// Constraint function applied to the `kernel` weights matrix.
        /// </summary>
        [JsonProperty("kernel_constraint")]
        public Action KernelConstraint { get; set; }

        /// <summary>
        /// Constraint function applied to the bias vector.
        /// </summary>
        [JsonProperty("bias_constraint")]
        public Action BiasConstraint { get; set; }

        [JsonProperty("name")]
        public override string Name { get => base.Name; set => base.Name = value; }
        [JsonProperty("dtype")]
        public override TF_DataType DType { get => base.DType; set => base.DType = value; }
        [JsonProperty("trainable")]
        public override bool Trainable { get => base.Trainable; set => base.Trainable = value; }
    }
}
