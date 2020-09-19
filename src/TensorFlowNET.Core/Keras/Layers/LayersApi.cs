using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class LayersApi
    {
        public Conv2D Conv2D(int filters,
            TensorShape kernel_size = null,
            string padding = "valid",
            string activation = "relu")
            => new Conv2D(new Conv2DArgs
            {
                Filters = filters,
                KernelSize = kernel_size,
                Padding = padding,
                Activation = GetActivationByName(activation)
            });


        public Dense Dense(int units,
            string activation = "linear",
            TensorShape input_shape = null)
            => new Dense(new DenseArgs
            {
                Units = units,
                Activation = GetActivationByName(activation),
                InputShape = input_shape
            });

        /// <summary>
        /// Turns positive integers (indexes) into dense vectors of fixed size.
        /// This layer can only be used as the first layer in a model.
        /// e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        /// </summary>
        /// <param name="input_dim">Size of the vocabulary, i.e. maximum integer index + 1.</param>
        /// <param name="output_dim">Dimension of the dense embedding.</param>
        /// <param name="embeddings_initializer">Initializer for the embeddings matrix (see keras.initializers).</param>
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

        public Flatten Flatten(string data_format = null)
            => new Flatten(new FlattenArgs
            {
                DataFormat = data_format
            });

        public MaxPooling2D MaxPooling2D(TensorShape pool_size = null,
            TensorShape strides = null,
            string padding = "valid")
            => new MaxPooling2D(new MaxPooling2DArgs
            {
                PoolSize = pool_size ?? (2, 2),
                Strides = strides,
                Padding = padding
            });

        public Rescaling Rescaling(float scale,
            float offset = 0,
            TensorShape input_shape = null)
            => new Rescaling(new RescalingArgs
            {
                Scale = scale,
                Offset = offset,
                InputShape = input_shape
            });

        Activation GetActivationByName(string name)
            => name switch
            {
                "linear" => tf.keras.activations.Linear,
                "relu" => tf.keras.activations.Relu,
                "sigmoid" => tf.keras.activations.Sigmoid,
                "tanh" => tf.keras.activations.Tanh,
                _ => tf.keras.activations.Linear
            };
    }
}
