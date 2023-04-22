using Newtonsoft.Json;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.ArgsDefinition
{
    [JsonObject(MemberSerialization.OptIn)]
    public class LayerArgs: IKerasConfig
    {
        /// <summary>
        /// Indicates whether the layer's weights are updated during training
        /// and whether the layer's updates are run during training.
        /// </summary>
        public virtual bool Trainable { get; set; } = true;
        public virtual string Name { get; set; }

        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        public virtual TF_DataType DType { get; set; } = TF_DataType.TF_FLOAT;

        /// <summary>
        /// Whether the `call` method can be used to build a TF graph without issues.
        /// This attribute has no effect if the model is created using the Functional
        /// API. Instead, `model.dynamic` is determined based on the internal layers.
        /// </summary>
        public virtual bool Dynamic { get; set; } = false;

        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        public virtual Shape InputShape { get; set; }

        /// <summary>
        /// Only applicable to input layers.
        /// </summary>
        public virtual KerasShapesWrapper BatchInputShape { get; set; }

        public virtual int BatchSize { get; set; } = -1;

        /// <summary>
        /// Initial weight values.
        /// </summary>
        public virtual float[] Weights { get; set; }

        /// <summary>
        /// Regularizer function applied to the output of the layer(its "activation").
        /// </summary>
        public virtual IRegularizer ActivityRegularizer { get; set; }

        public virtual bool Autocast { get; set; }

        public virtual bool IsFromConfig { get; set; }
    }
}
