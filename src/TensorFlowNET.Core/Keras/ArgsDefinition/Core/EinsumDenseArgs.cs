using Newtonsoft.Json;
using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition.Core
{
    public class EinsumDenseArgs : AutoSerializeLayerArgs
    {
        /// <summary>
        /// An equation describing the einsum to perform. This equation must
        /// be a valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
        /// `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum axis
        /// expression sequence.
        /// </summary>
        [JsonProperty("equation")]
        public string Equation { get; set; }

        /// <summary>
        /// The expected shape of the output tensor (excluding the batch
        /// dimension and any dimensions represented by ellipses). You can specify
        /// None for any dimension that is unknown or can be inferred from the input
        /// shape.
        /// </summary>
        [JsonProperty("output_shape")]
        public Shape OutputShape { get; set; }

        /// <summary>
        /// A string containing the output dimension(s) to apply a bias to.
        /// Each character in the `bias_axes` string should correspond to a character
        /// in the output portion of the `equation` string.
        /// </summary>
        [JsonProperty("bias_axes")]
        public string BiasAxes { get; set; } = null;

        /// <summary>
        /// Activation function to use.
        /// </summary>
        [JsonProperty("activation")]
        public Activation Activation { get; set; }

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
        [JsonProperty("activity_regularizer")]
        public override IRegularizer ActivityRegularizer { get => base.ActivityRegularizer; set => base.ActivityRegularizer = value; }
    }
}
