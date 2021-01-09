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
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras
{
    public class KerasInterface
    {
        public KerasDataset datasets { get; } = new KerasDataset();
        public Initializers initializers { get; } = new Initializers();
        public Regularizers regularizers { get; } = new Regularizers();
        public LayersApi layers { get; } = new LayersApi();
        public LossesApi losses { get; } = new LossesApi();
        public Activations activations { get; } = new Activations();
        public Preprocessing preprocessing { get; } = new Preprocessing();
        public BackendImpl backend { get; } = new BackendImpl();
        public OptimizerApi optimizers { get; } = new OptimizerApi();
        public MetricsApi metrics { get; } = new MetricsApi();
        public ModelsApi models { get; } = new ModelsApi();

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
        public Tensor Input(TensorShape shape = null,
                int batch_size = -1,
                TensorShape batch_input_shape = null,
                TF_DataType dtype = TF_DataType.DtInvalid,
                string name = null,
                bool sparse = false,
                bool ragged = false,
                Tensor tensor = null)
        {
            if (batch_input_shape != null)
                shape = batch_input_shape.dims[1..];

            var args = new InputLayerArgs
            {
                Name = name,
                InputShape = shape,
                BatchInputShape = batch_input_shape,
                BatchSize = batch_size,
                DType = dtype,
                Sparse = sparse,
                Ragged = ragged,
                InputTensor = tensor
            };

            var layer = new InputLayer(args);

            return layer.InboundNodes[0].Outputs;
        }
    }
}
