using System;
using System.Data;
using System.Linq;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Datasets;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class KerasApi
    {
        public KerasDataset datasets { get; } = new KerasDataset();
        public Initializers initializers { get; } = new Initializers();
        public LayersApi layers { get; } = new LayersApi();
        public Activations activations { get; } = new Activations();

        public BackendImpl backend { get; } = new BackendImpl();

        public Models models { get; } = new Models();

        public Sequential Sequential() 
            => new Sequential();

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
                TF_DataType dtype = TF_DataType.DtInvalid,
                string name = null,
                bool sparse = false,
                bool ragged = false,
                Tensor tensor = null)
        {
            var args = new InputLayerArgs
            {
                Name = name,
                InputShape = shape,
                BatchSize = batch_size,
                DType = dtype,
                Sparse = sparse,
                Ragged = ragged,
                InputTensor = tensor
            };

            var layer = new InputLayer(args);

            return layer.InboundNodes[0].Outputs;
        }

        public class LayersApi
        {
            public Layer Dense(int units,
                Activation activation = null,
                TensorShape input_shape = null)
                => new Dense(new DenseArgs
                {
                    Units = units,
                    Activation = activation ?? tf.keras.activations.Linear,
                    InputShape = input_shape
                });

            /// <summary>
            /// Turns positive integers (indexes) into dense vectors of fixed size.
            /// </summary>
            /// <param name="input_dim"></param>
            /// <param name="output_dim"></param>
            /// <param name="embeddings_initializer"></param>
            /// <param name="mask_zero"></param>
            /// <returns></returns>
            public Embedding Embedding(int input_dim,
                int output_dim,
                IInitializer embeddings_initializer = null,
                bool mask_zero = false,
                TensorShape input_shape = null,
                int input_length = -1)
                => new Embedding(new EmbeddingArgs
                {
                    InputDim = input_dim,
                    OutputDim = output_dim,
                    MaskZero = mask_zero,
                    InputShape = input_shape ?? input_length,
                    InputLength = input_length,
                    EmbeddingsInitializer = embeddings_initializer
                });
        }
    }
}
