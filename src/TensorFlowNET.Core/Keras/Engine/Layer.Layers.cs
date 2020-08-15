using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        protected List<Layer> _layers = new List<Layer>();

        protected Layer Dense(int units,
            Activation activation = null,
            TensorShape input_shape = null)
        {
            var layer = new Dense(new DenseArgs
            {
                Units = units,
                Activation = activation ?? tf.keras.activations.Linear,
                InputShape = input_shape
            });

            _layers.Add(layer);
            return layer;
        }

        protected Layer Conv2D(int filters,
            int kernel_size,
            TensorShape strides = null,
            string padding = "valid",
            string data_format = null,
            TensorShape dilation_rate = null,
            int groups = 1,
            Activation activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null,
            bool trainable = true,
            string name = null)
        {
            var layer = new Conv2D(new Conv2DArgs
            {
                Filters = filters,
                KernelSize = kernel_size,
                Strides = strides ?? (1, 1),
                Padding = padding,
                DataFormat = data_format,
                DilationRate = dilation_rate ?? (1, 1),
                Groups = groups,
                Activation = activation,
                UseBias = use_bias,
                KernelInitializer = kernel_initializer ?? tf.glorot_uniform_initializer,
                BiasInitializer = bias_initializer ?? tf.zeros_initializer,
                Trainable = trainable,
                Name = name
            });

            _layers.Add(layer);
            return layer;
        }

        protected Layer MaxPooling2D(TensorShape pool_size,
            TensorShape strides,
            string padding = "valid",
            string data_format = null,
            string name = null)
        {
            var layer = new MaxPooling2D(new MaxPooling2DArgs
            {
                PoolSize = pool_size,
                Strides = strides,
                Padding = padding,
                DataFormat = data_format,
                Name = name
            });

            _layers.Add(layer);
            return layer;
        }

        protected Layer Dropout(float rate, TensorShape noise_shape = null, int? seed = null)
        {
            var layer = new Dropout(new DropoutArgs
            {
                Rate = rate,
                NoiseShape = noise_shape,
                Seed = seed
            });

            _layers.Add(layer);
            return layer;
        }

        protected Layer Flatten()
        {
            var layer = new Flatten(new FlattenArgs());

            _layers.Add(layer);
            return layer;
        }
    }
}
