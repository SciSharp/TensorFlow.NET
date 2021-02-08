using NumSharp;
using System.Collections.Generic;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers
{
    public partial class LayersApi
    {
        /// <summary>
        /// Functional interface for the batch normalization layer.
        /// http://arxiv.org/abs/1502.03167
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="axis"></param>
        /// <param name="momentum"></param>
        /// <param name="epsilon"></param>
        /// <param name="center"></param>
        /// <param name="scale"></param>
        /// <param name="beta_initializer"></param>
        /// <param name="gamma_initializer"></param>
        /// <param name="moving_mean_initializer"></param>
        /// <param name="moving_variance_initializer"></param>
        /// <param name="training"></param>
        /// <param name="trainable"></param>
        /// <param name="name"></param>
        /// <param name="renorm"></param>
        /// <param name="renorm_momentum"></param>
        /// <returns></returns>
        public BatchNormalization BatchNormalization(int axis = -1,
            float momentum = 0.99f,
            float epsilon = 0.001f,
            bool center = true,
            bool scale = true,
            IInitializer beta_initializer = null,
            IInitializer gamma_initializer = null,
            IInitializer moving_mean_initializer = null,
            IInitializer moving_variance_initializer = null,
            bool trainable = true,
            string name = null,
            bool renorm = false,
            float renorm_momentum = 0.99f)
                => new BatchNormalization(new BatchNormalizationArgs
                {
                    Axis = axis,
                    Momentum = momentum,
                    Epsilon = epsilon,
                    Center = center,
                    Scale = scale,
                    BetaInitializer = beta_initializer ?? tf.zeros_initializer,
                    GammaInitializer = gamma_initializer ?? tf.ones_initializer,
                    MovingMeanInitializer = moving_mean_initializer ?? tf.zeros_initializer,
                    MovingVarianceInitializer = moving_variance_initializer ?? tf.ones_initializer,
                    Renorm = renorm,
                    RenormMomentum = renorm_momentum,
                    Trainable = trainable,
                    Name = name
                });

        /// <summary>
        /// 
        /// </summary>
        /// <param name="filters"></param>
        /// <param name="kernel_size"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="data_format"></param>
        /// <param name="dilation_rate"></param>
        /// <param name="groups"></param>
        /// <param name="activation">tf.keras.activations</param>
        /// <param name="use_bias"></param>
        /// <param name="kernel_initializer"></param>
        /// <param name="bias_initializer"></param>
        /// <param name="kernel_regularizer"></param>
        /// <param name="bias_regularizer"></param>
        /// <param name="activity_regularizer"></param>
        /// <returns></returns>
        public Conv2D Conv2D(int filters,
            TensorShape kernel_size = null,
            TensorShape strides = null,
            string padding = "valid",
            string data_format = null,
            TensorShape dilation_rate = null,
            int groups = 1,
            Activation activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null,
            IRegularizer kernel_regularizer = null,
            IRegularizer bias_regularizer = null,
            IRegularizer activity_regularizer = null)
                => new Conv2D(new Conv2DArgs
                {
                    Rank = 2,
                    Filters = filters,
                    KernelSize = kernel_size,
                    Strides = strides == null ? (1, 1) : strides,
                    Padding = padding,
                    DataFormat = data_format,
                    DilationRate = dilation_rate == null ? (1, 1) : dilation_rate,
                    Groups = groups,
                    UseBias = use_bias,
                    KernelRegularizer = kernel_regularizer,
                    KernelInitializer = kernel_initializer == null ? tf.glorot_uniform_initializer : kernel_initializer,
                    BiasInitializer = bias_initializer == null ? tf.zeros_initializer : bias_initializer,
                    BiasRegularizer = bias_regularizer,
                    ActivityRegularizer = activity_regularizer,
                    Activation = activation ?? keras.activations.Linear
                });

        public Conv2D Conv2D(int filters,
            TensorShape kernel_size = null,
            TensorShape strides = null,
            string padding = "valid",
            string data_format = null,
            TensorShape dilation_rate = null,
            int groups = 1,
            string activation = null,
            bool use_bias = true,
            string kernel_initializer = "glorot_uniform",
            string bias_initializer = "zeros",
            string kernel_regularizer = null,
            string bias_regularizer = null,
            string activity_regularizer = null)
                => new Conv2D(new Conv2DArgs
                {
                    Rank = 2,
                    Filters = filters,
                    KernelSize = kernel_size,
                    Strides = strides == null ? (1, 1) : strides,
                    Padding = padding,
                    DataFormat = data_format,
                    DilationRate = dilation_rate == null ? (1, 1) : dilation_rate,
                    Groups = groups,
                    UseBias = use_bias,
                    KernelInitializer = GetInitializerByName(kernel_initializer),
                    BiasInitializer = GetInitializerByName(bias_initializer),
                    Activation = GetActivationByName(activation)
                });

        /// <summary>
        /// Transposed convolution layer (sometimes called Deconvolution).
        /// </summary>
        /// <param name="filters"></param>
        /// <param name="kernel_size"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="data_format"></param>
        /// <param name="dilation_rate"></param>
        /// <param name="activation"></param>
        /// <param name="use_bias"></param>
        /// <param name="kernel_initializer"></param>
        /// <param name="bias_initializer"></param>
        /// <param name="kernel_regularizer"></param>
        /// <param name="bias_regularizer"></param>
        /// <param name="activity_regularizer"></param>
        /// <returns></returns>
        public Conv2DTranspose Conv2DTranspose(int filters,
            TensorShape kernel_size = null,
            TensorShape strides = null,
            string padding = "valid",
            string data_format = null,
            TensorShape dilation_rate = null,
            string activation = null,
            bool use_bias = true,
            string kernel_initializer = null,
            string bias_initializer = null,
            string kernel_regularizer = null,
            string bias_regularizer = null,
            string activity_regularizer = null)
                => new Conv2DTranspose(new Conv2DArgs
                {
                    Rank = 2,
                    Filters = filters,
                    KernelSize = kernel_size,
                    Strides = strides == null ? (1, 1) : strides,
                    Padding = padding,
                    DataFormat = data_format,
                    DilationRate = dilation_rate == null ? (1, 1) : dilation_rate,
                    UseBias = use_bias,
                    KernelInitializer = GetInitializerByName(kernel_initializer),
                    BiasInitializer = GetInitializerByName(bias_initializer),
                    Activation = GetActivationByName(activation)
                });

        public Dense Dense(int units,
            Activation activation = null,
            IInitializer kernel_initializer = null,
            bool use_bias = true,
            IInitializer bias_initializer = null,
            TensorShape input_shape = null)
            => new Dense(new DenseArgs
            {
                Units = units,
                Activation = activation ?? keras.activations.Linear,
                KernelInitializer = kernel_initializer ?? tf.glorot_uniform_initializer,
                BiasInitializer = bias_initializer ?? (use_bias ? tf.zeros_initializer : null),
                InputShape = input_shape
            });

        public Dense Dense(int units)
            => new Dense(new DenseArgs
            {
                Units = units,
                Activation = GetActivationByName("linear")
            });

        public Dense Dense(int units,
            string activation = null,
            TensorShape input_shape = null)
            => new Dense(new DenseArgs
            {
                Units = units,
                Activation = GetActivationByName(activation),
                InputShape = input_shape
            });

        /// <summary>
        ///     Densely-connected layer class. aka fully-connected<br></br>
        ///     `outputs = activation(inputs * kernel + bias)`
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="units">Python integer, dimensionality of the output space.</param>
        /// <param name="activation"></param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias.</param>
        /// <param name="kernel_initializer"></param>
        /// <param name="bias_initializer"></param>
        /// <param name="trainable"></param>
        /// <param name="name"></param>
        /// <param name="reuse"></param>
        /// <returns></returns>
        public Tensor dense(Tensor inputs,
            int units,
            Activation activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null,
            bool trainable = true,
            string name = null,
            bool? reuse = null)
        {
            if (bias_initializer == null)
                bias_initializer = tf.zeros_initializer;

            var layer = new Dense(new DenseArgs
            {
                Units = units,
                Activation = activation,
                UseBias = use_bias,
                BiasInitializer = bias_initializer,
                KernelInitializer = kernel_initializer,
                Trainable = trainable,
                Name = name
            });

            return layer.Apply(inputs);
        }

        public Dropout Dropout(float rate, TensorShape noise_shape = null, int? seed = null)
            => new Dropout(new DropoutArgs
            {
                Rate = rate,
                NoiseShape = noise_shape,
                Seed = seed
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

        /// <summary>
        /// `Input()` is used to instantiate a Keras tensor.
        /// </summary>
        /// <param name="shape">A shape tuple not including the batch size.</param>
        /// <param name="name"></param>
        /// <param name="sparse"></param>
        /// <param name="ragged"></param>
        /// <returns></returns>
        public Tensors Input(TensorShape shape,
            string name = null,
            bool sparse = false,
            bool ragged = false)
        {
            var input_layer = new InputLayer(new InputLayerArgs
            {
                InputShape = shape,
                Name = name,
                Sparse = sparse,
                Ragged = ragged
            });

            return input_layer.InboundNodes[0].Outputs;
        }

        public MaxPooling2D MaxPooling2D(TensorShape pool_size = null,
            TensorShape strides = null,
            string padding = "valid")
            => new MaxPooling2D(new MaxPooling2DArgs
            {
                PoolSize = pool_size ?? (2, 2),
                Strides = strides,
                Padding = padding
            });

        /// <summary>
        /// Max pooling layer for 2D inputs (e.g. images).
        /// </summary>
        /// <param name="inputs">The tensor over which to pool. Must have rank 4.</param>
        /// <param name="pool_size"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="data_format"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor max_pooling2d(Tensor inputs,
            int[] pool_size,
            int[] strides,
            string padding = "valid",
            string data_format = "channels_last",
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

            return layer.Apply(inputs);
        }

        /// <summary>
        /// Leaky version of a Rectified Linear Unit.
        /// </summary>
        /// <param name="alpha">Negative slope coefficient.</param>
        /// <returns></returns>
        public Layer LeakyReLU(float alpha = 0.3f)
            => new LeakyReLu(new LeakyReLuArgs
            {
                Alpha = alpha
            });

        public Layer SimpleRNN(int units) => SimpleRNN(units, "tanh");

        public Layer SimpleRNN(int units,
            Activation activation = null)
                => new SimpleRNN(new SimpleRNNArgs
                {
                    Units = units,
                    Activation = activation
                });

        public Layer SimpleRNN(int units,
            string activation = "tanh")
                => new SimpleRNN(new SimpleRNNArgs
                {
                    Units = units,
                    Activation = GetActivationByName(activation)
                });

        public Layer LSTM(int units,
            Activation activation = null,
            Activation recurrent_activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer recurrent_initializer = null,
            IInitializer bias_initializer = null,
            bool unit_forget_bias = true,
            float dropout = 0f,
            float recurrent_dropout = 0f,
            int implementation = 2,
            bool return_sequences = false,
            bool return_state = false,
            bool go_backwards = false,
            bool stateful = false,
            bool time_major = false,
            bool unroll = false)
                => new LSTM(new LSTMArgs
                {
                    Units = units,
                    Activation = activation ?? keras.activations.Tanh,
                    RecurrentActivation = recurrent_activation ?? keras.activations.Sigmoid,
                    KernelInitializer = kernel_initializer ?? tf.glorot_uniform_initializer,
                    RecurrentInitializer = recurrent_initializer ?? tf.orthogonal_initializer,
                    BiasInitializer = bias_initializer ?? tf.zeros_initializer,
                    Dropout = dropout,
                    RecurrentDropout = recurrent_dropout,
                    Implementation = implementation,
                    ReturnSequences = return_sequences,
                    ReturnState = return_state,
                    GoBackwards = go_backwards,
                    Stateful = stateful,
                    TimeMajor = time_major,
                    Unroll = unroll
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

        public Add Add()
            => new Add(new MergeArgs { });

        public Subtract Subtract()
            => new Subtract(new MergeArgs { });

        public GlobalAveragePooling2D GlobalAveragePooling2D()
            => new GlobalAveragePooling2D(new Pooling2DArgs { });

        Activation GetActivationByName(string name)
            => name switch
            {
                "linear" => keras.activations.Linear,
                "relu" => keras.activations.Relu,
                "sigmoid" => keras.activations.Sigmoid,
                "tanh" => keras.activations.Tanh,
                _ => keras.activations.Linear
            };

        IInitializer GetInitializerByName(string name)
            => name switch
            {
                "glorot_uniform" => tf.glorot_uniform_initializer,
                "zeros" => tf.zeros_initializer,
                "ones" => tf.ones_initializer,
                _ => tf.glorot_uniform_initializer
            };
    }
}
