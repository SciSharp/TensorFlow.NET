using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class ConvolutionalArgs : LayerArgs
    {
        public int Rank { get; set; } = 2;
        public int Filters { get; set; }
        public int NumSpatialDims { get; set; } = Unknown;
        public TensorShape KernelSize { get; set; } = 5;

        /// <summary>
        /// specifying the stride length of the convolution.
        /// </summary>
        public TensorShape Strides { get; set; } = (1, 1);

        public string Padding { get; set; } = "valid";
        public string DataFormat { get; set; }
        public TensorShape DilationRate { get; set; } = (1, 1);
        public int Groups { get; set; } = 1;
        public Activation Activation { get; set; }
        public bool UseBias { get; set; }
        public IInitializer KernelInitializer { get; set; } = tf.glorot_uniform_initializer;
        public IInitializer BiasInitializer { get; set; } = tf.zeros_initializer;
        public IRegularizer KernelRegularizer { get; set; }
        public IRegularizer BiasRegularizer { get; set; }
        public Action KernelConstraint { get; set; }
        public Action BiasConstraint { get; set; }
    }
}
