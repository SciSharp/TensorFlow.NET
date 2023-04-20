using Newtonsoft.Json;
using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class ConvolutionalArgs : AutoSerializeLayerArgs
    {
        public int Rank { get; set; }
        [JsonProperty("filters")]
        public int Filters { get; set; }
        public int NumSpatialDims { get; set; } = Unknown;
        [JsonProperty("kernel_size")]
        public Shape KernelSize { get; set; }

        /// <summary>
        /// specifying the stride length of the convolution.
        /// </summary>
        [JsonProperty("strides")]
        public Shape Strides { get; set; }
        [JsonProperty("padding")]
        public string Padding { get; set; }
        [JsonProperty("data_format")]
        public string DataFormat { get; set; }
        [JsonProperty("dilation_rate")]
        public Shape DilationRate { get; set; }
        [JsonProperty("groups")]
        public int Groups { get; set; }
        [JsonProperty("activation")]
        public Activation Activation { get; set; }
        [JsonProperty("use_bias")]
        public bool UseBias { get; set; }
        [JsonProperty("kernel_initializer")]
        public IInitializer KernelInitializer { get; set; }
        [JsonProperty("bias_initializer")]
        public IInitializer BiasInitializer { get; set; }
        [JsonProperty("kernel_regularizer")]
        public IRegularizer KernelRegularizer { get; set; }
        [JsonProperty("bias_regularizer")]
        public IRegularizer BiasRegularizer { get; set; }
        [JsonProperty("kernel_constraint")]
        public Action KernelConstraint { get; set; }
        [JsonProperty("bias_constraint")]
        public Action BiasConstraint { get; set; }
    }
}
