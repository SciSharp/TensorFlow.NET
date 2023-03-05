using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using Tensorflow.Framework.Models;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;
using Tensorflow.Keras.Models;

namespace Tensorflow.Keras
{
    public interface IKerasApi
    {
        IInitializersApi initializers { get; }
        ILayersApi layers { get; }
        ILossesApi losses { get; }
        IActivationsApi activations { get; }
        IOptimizerApi optimizers { get; }
        IMetricsApi metrics { get; }
        IModelsApi models { get; }

        /// <summary>
        /// `Model` groups layers into an object with training and inference features.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        IModel Model(Tensors inputs, Tensors outputs, string name = null);

        /// <summary>
        /// Instantiate a Keras tensor.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="batch_size"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <param name="sparse">
        /// A boolean specifying whether the placeholder to be created is sparse.
        /// </param>
        /// <param name="ragged">
        /// A boolean specifying whether the placeholder to be created is ragged.
        /// </param>
        /// <param name="tensor">
        /// Optional existing tensor to wrap into the `Input` layer.
        /// If set, the layer will not create a placeholder tensor.
        /// </param>
        /// <returns></returns>
        Tensors Input(Shape shape = null,
            int batch_size = -1,
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            bool sparse = false,
            Tensor tensor = null,
            bool ragged = false,
            TypeSpec type_spec = null,
            Shape batch_input_shape = null,
            Shape batch_shape = null);
    }
}
