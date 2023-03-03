using System;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Datasets;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Utils;
using System.Threading;
using Tensorflow.Framework.Models;

namespace Tensorflow.Keras
{
    public class KerasInterface : IKerasApi
    {
        public KerasDataset datasets { get; } = new KerasDataset();
        public IInitializersApi initializers { get; } = new InitializersApi();
        public Regularizers regularizers { get; } = new Regularizers();
        public ILayersApi layers { get; } = new LayersApi();
        public ILossesApi losses { get; } = new LossesApi();
        public Activations activations { get; } = new Activations();
        public Preprocessing preprocessing { get; } = new Preprocessing();
        ThreadLocal<BackendImpl> _backend = new ThreadLocal<BackendImpl>(() => new BackendImpl());
        public BackendImpl backend => _backend.Value;
        public OptimizerApi optimizers { get; } = new OptimizerApi();
        public IMetricsApi metrics { get; } = new MetricsApi();
        public ModelsApi models { get; } = new ModelsApi();
        public KerasUtils utils { get; } = new KerasUtils();

        public Sequential Sequential(List<ILayer> layers = null,
                string name = null)
            => new Sequential(new SequentialArgs
            {
                Layers = layers,
                Name = name
            });

        /// <summary>
        /// `Model` groups layers into an object with training and inference features.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        public Functional Model(Tensors inputs, Tensors outputs, string name = null)
            => new Functional(inputs, outputs, name: name);

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
        public Tensors Input(Shape shape = null,
            int batch_size = -1,
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            bool sparse = false,
            Tensor tensor = null,
            bool ragged = false,
            TypeSpec type_spec = null,
            Shape batch_input_shape = null,
            Shape batch_shape = null) => keras.layers.Input(shape, batch_size, name,
                dtype, sparse, tensor, ragged, type_spec, batch_input_shape, batch_shape);
    }
}
