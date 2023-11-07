using System;
using Tensorflow.Framework.Models;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.NumPy;
using static Google.Protobuf.Reflection.FieldDescriptorProto.Types;

namespace Tensorflow.Keras.Layers
{
    public partial interface ILayersApi
    {
        public IPreprocessing preprocessing { get; }

        public ILayer Add();

        public ILayer AveragePooling2D(Shape pool_size = null,
            Shape strides = null,
            string padding = "valid",
            string data_format = null);

        public ILayer BatchNormalization(int axis = -1,
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
            float renorm_momentum = 0.99f);

        /// <summary>
        /// A preprocessing layer which encodes integer features.
        /// </summary>
        /// <param name="num_tokens">The total number of tokens the layer should support.</param>
        /// <param name="output_mode">Specification for the output of the layer.</param>
        /// <returns></returns>
        public ILayer CategoryEncoding(int num_tokens, 
            string output_mode = "one_hot",
            bool sparse = false,
            NDArray count_weights = null);

        public ILayer Conv1D(int filters,
            Shape kernel_size,
            int strides = 1,
            string padding = "valid",
            string data_format = "channels_last",
            int dilation_rate = 1,
            int groups = 1,
            string activation = null,
            bool use_bias = true,
            string kernel_initializer = "glorot_uniform",
            string bias_initializer = "zeros");

        public ILayer Conv2D(int filters,
                Shape kernel_size = null,
                Shape strides = null,
                string padding = "valid"
            );

        public ILayer Conv2D(int filters,
            Shape kernel_size = null,
            Shape strides = null,
            string padding = "valid",
            string data_format = null,
            Shape dilation_rate = null,
            int groups = 1,
            Activation activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null,
            IRegularizer kernel_regularizer = null,
            IRegularizer bias_regularizer = null,
            IRegularizer activity_regularizer = null);

        public ILayer Conv2DTranspose(int filters,
            Shape kernel_size = null,
            Shape strides = null,
            string output_padding = "valid",
            string data_format = null,
            Shape dilation_rate = null,
            string activation = null,
            bool use_bias = true,
            string kernel_initializer = null,
            string bias_initializer = null,
            string kernel_regularizer = null,
            string bias_regularizer = null,
            string activity_regularizer = null);

        public ILayer Conv2D(int filters,
            Shape kernel_size = null,
            Shape strides = null,
            string padding = "valid",
            string data_format = null,
            Shape dilation_rate = null,
            int groups = 1,
            string activation = null,
            bool use_bias = true,
            string kernel_initializer = "glorot_uniform",
            string bias_initializer = "zeros");
        public ILayer DepthwiseConv2D(Shape kernel_size = null,
            Shape strides = null,
            string padding = "valid",
            string data_format = null,
            Shape dilation_rate = null,
            int groups = 1,
            int depth_multiplier = 1,
            string activation = null,
            bool use_bias = false,
            string kernel_initializer = "glorot_uniform",
            string bias_initializer = "zeros",
            string depthwise_initializer = "glorot_uniform"
            );

        public ILayer Dense(int units);
        public ILayer Dense(int units,
            string activation = null,
            Shape input_shape = null);
        public ILayer Dense(int units,
            Activation activation = null,
            IInitializer kernel_initializer = null,
            bool use_bias = true,
            IInitializer bias_initializer = null,
            Shape input_shape = null);

        public ILayer Dropout(float rate, Shape noise_shape = null, int? seed = null);

        public ILayer Embedding(int input_dim,
            int output_dim,
            IInitializer embeddings_initializer = null,
            bool mask_zero = false,
            Shape input_shape = null,
            int input_length = -1);

        public ILayer EinsumDense(string equation,
                Shape output_shape,
                string bias_axes,
                Activation activation = null,
                IInitializer kernel_initializer = null,
                IInitializer bias_initializer = null,
                IRegularizer kernel_regularizer = null,
                IRegularizer bias_regularizer = null,
                IRegularizer activity_regularizer = null,
                Action kernel_constraint = null,
                Action bias_constraint = null);

        public ILayer Flatten(string data_format = null);

        public ILayer GlobalAveragePooling1D(string data_format = "channels_last");
        public ILayer GlobalAveragePooling2D();
        public ILayer GlobalAveragePooling2D(string data_format = "channels_last");
        public ILayer GlobalMaxPooling1D(string data_format = "channels_last");
        public ILayer GlobalMaxPooling2D(string data_format = "channels_last");

        public KerasTensor Input(Shape shape = null,
            int batch_size = -1,
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            bool sparse = false,
            Tensor tensor = null,
            bool ragged = false,
            TypeSpec type_spec = null,
            Shape batch_input_shape = null,
            Shape batch_shape = null);
        public ILayer InputLayer(Shape input_shape,
            string name = null,
            bool sparse = false,
            bool ragged = false);

        public ILayer LayerNormalization(Axis? axis,
           float epsilon = 1e-3f,
           bool center = true,
           bool scale = true,
           IInitializer beta_initializer = null,
           IInitializer gamma_initializer = null);

        public ILayer Normalization(Shape? input_shape = null, int? axis = -1, float? mean = null, float? variance = null, bool invert = false);
        public ILayer LeakyReLU(float alpha = 0.3f);

        public ILayer ReLU6();


        public IRnnCell LSTMCell(int uints,
            string activation = "tanh",
            string recurrent_activation = "sigmoid",
            bool use_bias = true,
            string kernel_initializer = "glorot_uniform",
            string recurrent_initializer = "orthogonal",
            string bias_initializer = "zeros",
            bool unit_forget_bias = true,
            float dropout = 0f,
            float recurrent_dropout = 0f,
            int implementation = 2);

        public ILayer LSTM(int units,
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
            bool unroll = false);

        public ILayer MaxPooling1D(int? pool_size = null,
            int? strides = null,
            string padding = "valid",
            string data_format = null);
        public ILayer MaxPooling2D(Shape pool_size = null,
            Shape strides = null,
            string padding = "valid",
            string data_format = null);

        public ILayer Permute(int[] dims);

        public ILayer Rescaling(float scale,
            float offset = 0,
            Shape input_shape = null);

        public IRnnCell SimpleRNNCell(
            int units,
            string activation = "tanh",
            bool use_bias = true,
            string kernel_initializer = "glorot_uniform",
            string recurrent_initializer = "orthogonal",
            string bias_initializer = "zeros",
            float dropout = 0f,
            float recurrent_dropout = 0f);

        public IRnnCell StackedRNNCells(
            IEnumerable<IRnnCell> cells);

        public ILayer SimpleRNN(int units,
            string activation = "tanh",
            string kernel_initializer = "glorot_uniform",
            string recurrent_initializer = "orthogonal",
            string bias_initializer = "zeros",
            bool return_sequences = false,
            bool return_state = false);

        public ILayer RNN(
            IRnnCell cell,
            bool return_sequences = false,
            bool return_state = false,
            bool go_backwards = false,
            bool stateful = false,
            bool unroll = false,
            bool time_major = false
            );

        public ILayer RNN(
            IEnumerable<IRnnCell> cell,
            bool return_sequences = false,
            bool return_state = false,
            bool go_backwards = false,
            bool stateful = false,
            bool unroll = false,
            bool time_major = false
            );

        public IRnnCell GRUCell(
            int units,
            string activation = "tanh",
            string recurrent_activation = "sigmoid",
            bool use_bias = true,
            string kernel_initializer = "glorot_uniform",
            string recurrent_initializer = "orthogonal",
            string bias_initializer = "zeros",
            float dropout = 0f,
            float recurrent_dropout = 0f, 
            bool reset_after = true);

        public ILayer GRU(
            int units,
            string activation = "tanh",
            string recurrent_activation = "sigmoid",
            bool use_bias = true,
            string kernel_initializer = "glorot_uniform",
            string recurrent_initializer = "orthogonal",
            string bias_initializer = "zeros",
            float dropout = 0f,
            float recurrent_dropout = 0f,
            bool return_sequences = false,
            bool return_state = false,
            bool go_backwards = false,
            bool stateful = false,
            bool unroll = false,
            bool time_major = false,
            bool reset_after = true
            );

        /// <summary>
        /// Bidirectional wrapper for RNNs.
        /// </summary>
        /// <param name="layer">`keras.layers.RNN` instance, such as `keras.layers.LSTM` or `keras.layers.GRU`</param>
        /// automatically.</param>
        /// <returns></returns>
        public ILayer Bidirectional(
                ILayer layer,
                string merge_mode = "concat",
                NDArray weights = null,
                ILayer backward_layer = null);

        public ILayer Subtract();
    }
}
