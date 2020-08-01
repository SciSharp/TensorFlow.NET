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
        public Layers layers { get; } = new Layers();
        public Activations activations { get; } = new Activations();

        public BackendImpl backend { get; } = new BackendImpl();

        public Sequential Sequential() 
            => new Sequential();

        public Tensor[] Input(int[] batch_shape = null,
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
                BatchInputShape = batch_shape,
                BatchSize = batch_size,
                DType = dtype,
                Sparse = sparse,
                Ragged = ragged,
                InputTensor = tensor
            };

            var layer = new InputLayer(args);

            return layer.InboundNodes[0].Outputs;
        }

        public static Embedding Embedding(int input_dim,
            int output_dim,
            IInitializer embeddings_initializer = null,
            bool mask_zero = false)
            => new Embedding(input_dim,
                output_dim,
                embeddings_initializer,
                mask_zero);

        public class Layers
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
        }
    }
}
