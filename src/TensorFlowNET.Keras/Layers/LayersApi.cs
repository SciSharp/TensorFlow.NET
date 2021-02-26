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
        public Preprocessing preprocessing { get; } = new Preprocessing();

        /// <summary>
        /// Layer that normalizes its inputs.
        /// Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
        /// Importantly, batch normalization works differently during training and during inference.
        /// 
        /// http://arxiv.org/abs/1502.03167
        /// </summary>
        /// <param name="axis">The axis that should be normalized (typically the features axis). 
        /// For instance, after a Conv2D layer with data_format="channels_first", set axis=1 in BatchNormalization.
        /// </param>
        /// <param name="momentum">Momentum for the moving average.</param>
        /// <param name="epsilon">Small float added to variance to avoid dividing by zero.</param>
        /// <param name="center">If True, add offset of beta to normalized tensor. If False, beta is ignored.</param>
        /// <param name="scale">If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling will be done by the next layer.</param>
        /// <param name="beta_initializer">Initializer for the beta weight.</param>
        /// <param name="gamma_initializer">Initializer for the gamma weight.</param>
        /// <param name="moving_mean_initializer">Initializer for the moving mean.</param>
        /// <param name="moving_variance_initializer">Initializer for the moving variance.</param>
        /// <param name="trainable">Boolean, if True the variables will be marked as trainable.</param>
        /// <param name="name">Layer name.</param>
        /// <param name="renorm">Whether to use Batch Renormalization. This adds extra variables during training. The inference is the same for either value of this parameter.</param>
        /// <param name="renorm_momentum">Momentum used to update the moving means and standard deviations with renorm. 
        /// Unlike momentum, this affects training and should be neither too small (which would add noise) nor too large (which would give stale estimates). 
        /// Note that momentum is still applied to get the means and variances for inference.
        /// </param>
        /// <returns>Tensor of the same shape as input.</returns>
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
        /// 1D convolution layer (e.g. temporal convolution).
        /// This layer creates a convolution kernel that is convolved with the layer input over a single spatial(or temporal) dimension to produce a tensor of outputs.If use_bias is True, a bias vector is created and added to the outputs.Finally, if activation is not None, it is applied to the outputs as well.
        /// </summary>
        /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)</param>
        /// <param name="kernel_size">An integer specifying the width of the 1D convolution window.</param>
        /// <param name="strides">An integer specifying the stride of the convolution window . Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.</param>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last.</param>
        /// <param name="dilation_rate">An integer specifying the dilation rate to use for dilated convolution.Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="groups">A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix (see keras.initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see keras.initializers).</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see keras.regularizers).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see keras.regularizers).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation") (see keras.regularizers).</param>
        /// <returns>A tensor of rank 3 representing activation(conv1d(inputs, kernel) + bias).</returns>
        public Conv1D Conv1D(int filters,
            int? kernel_size = null,
            int? strides = null,
            string padding = "valid",
            string data_format = null,
            int? dilation_rate = null,
            int groups = 1,
            Activation activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null,
            IRegularizer kernel_regularizer = null,
            IRegularizer bias_regularizer = null,
            IRegularizer activity_regularizer = null)
        {
            // Special case: Conv1D will be implemented as Conv2D with H=1, so we need to add a 1-sized dimension to the kernel.
            // Lower-level logic handles the stride and dilation_rate, but the kernel_size needs to be set properly here.

            var kernel = (kernel_size == null) ? (1, 5) : (1, kernel_size.Value);
            return new Conv1D(new Conv1DArgs
            {
                Rank = 1,
                Filters = filters,
                KernelSize = kernel,
                Strides = strides == null ? 1 : strides,
                Padding = padding,
                DataFormat = data_format,
                DilationRate = dilation_rate == null ? 1 : dilation_rate,
                Groups = groups,
                UseBias = use_bias,
                KernelInitializer = kernel_initializer == null ? tf.glorot_uniform_initializer : kernel_initializer,
                BiasInitializer = bias_initializer == null ? tf.zeros_initializer : bias_initializer,
                KernelRegularizer = kernel_regularizer,
                BiasRegularizer = bias_regularizer,
                ActivityRegularizer = activity_regularizer,
                Activation = activation ?? keras.activations.Linear
            });
        }

        /// <summary>
        /// 1D convolution layer (e.g. temporal convolution).
        /// This layer creates a convolution kernel that is convolved with the layer input over a single spatial(or temporal) dimension to produce a tensor of outputs.If use_bias is True, a bias vector is created and added to the outputs.Finally, if activation is not None, it is applied to the outputs as well.
        /// </summary>
        /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)</param>
        /// <param name="kernel_size">An integer specifying the width of the 1D convolution window.</param>
        /// <param name="strides">An integer specifying the stride of the convolution window . Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.</param>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last.</param>
        /// <param name="dilation_rate">An integer specifying the dilation rate to use for dilated convolution.Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="groups">A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix (see keras.initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see keras.initializers).</param>
        /// <returns>A tensor of rank 3 representing activation(conv1d(inputs, kernel) + bias).</returns>
        public Conv1D Conv1D(int filters,
            int? kernel_size = null,
            int? strides = null,
            string padding = "valid",
            string data_format = null,
            int? dilation_rate = null,
            int groups = 1,
            string activation = null,
            bool use_bias = true,
            string kernel_initializer = "glorot_uniform",
            string bias_initializer = "zeros")
        {
            // Special case: Conv1D will be implemented as Conv2D with H=1, so we need to add a 1-sized dimension to the kernel.
            // Lower-level logic handles the stride and dilation_rate, but the kernel_size needs to be set properly here.

            var kernel = (kernel_size == null) ? (1, 5) : (1, kernel_size.Value);
            return new Conv1D(new Conv1DArgs
            {
                Rank = 1,
                Filters = filters,
                KernelSize = kernel,
                Strides = strides == null ? 1 : strides,
                Padding = padding,
                DataFormat = data_format,
                DilationRate = dilation_rate == null ? 1 : dilation_rate,
                Groups = groups,
                UseBias = use_bias,
                Activation = GetActivationByName(activation),
                KernelInitializer = GetInitializerByName(kernel_initializer),
                BiasInitializer = GetInitializerByName(bias_initializer)
            });
        }

        /// <summary>
        /// 2D convolution layer (e.g. spatial convolution over images).
        /// This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
        /// If use_bias is True, a bias vector is created and added to the outputs.Finally, if activation is not None, it is applied to the outputs as well.
        /// </summary>
        /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)</param>
        /// <param name="kernel_size">An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides">An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.</param>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last.</param>
        /// <param name="dilation_rate">an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="groups">A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix (see keras.initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see keras.initializers).</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see keras.regularizers).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see keras.regularizers).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation") (see keras.regularizers).</param>
        /// <returns>A tensor of rank 4+ representing activation(conv2d(inputs, kernel) + bias).</returns>
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
                    KernelSize = (kernel_size == null) ? (5, 5) : kernel_size,
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

        /// <summary>
        /// 2D convolution layer (e.g. spatial convolution over images).
        /// This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
        /// If use_bias is True, a bias vector is created and added to the outputs.Finally, if activation is not None, it is applied to the outputs as well.
        /// </summary>
        /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)</param>
        /// <param name="kernel_size">An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides">An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.</param>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last.</param>
        /// <param name="dilation_rate">an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="groups">A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">The name of the initializer for the kernel weights matrix (see keras.initializers).</param>
        /// <param name="bias_initializer">The name of the initializer for the bias vector (see keras.initializers).</param>
        /// <param name="kernel_regularizer">The name of the regularizer function applied to the kernel weights matrix (see keras.regularizers).</param>
        /// <param name="bias_regularizer">The name of the regularizer function applied to the bias vector (see keras.regularizers).</param>
        /// <param name="activity_regularizer">The name of the regularizer function applied to the output of the layer (its "activation") (see keras.regularizers).</param>
        /// <returns>A tensor of rank 4+ representing activation(conv2d(inputs, kernel) + bias).</returns>
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
            string bias_initializer = "zeros")
                => new Conv2D(new Conv2DArgs
                {
                    Rank = 2,
                    Filters = filters,
                    KernelSize = (kernel_size == null) ? (5,5) : kernel_size,
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
        /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)</param>
        /// <param name="kernel_size">An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.</param>
        /// <param name="strides">An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="output_padding">one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.</param>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last.</param>
        /// <param name="dilation_rate">an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">The name of the initializer for the kernel weights matrix (see keras.initializers).</param>
        /// <param name="bias_initializer">The name of the initializer for the bias vector (see keras.initializers).</param>
        /// <param name="kernel_regularizer">The name of the regularizer function applied to the kernel weights matrix (see keras.regularizers).</param>
        /// <param name="bias_regularizer">The name of the regularizer function applied to the bias vector (see keras.regularizers).</param>
        /// <param name="activity_regularizer">The name of the regularizer function applied to the output of the layer (its "activation") (see keras.regularizers).</param>
        /// <returns>A tensor of rank 4+ representing activation(conv2d(inputs, kernel) + bias).</returns>
        public Conv2DTranspose Conv2DTranspose(int filters,
            TensorShape kernel_size = null,
            TensorShape strides = null,
            string output_padding = "valid",
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
                    KernelSize = (kernel_size == null) ? (5, 5) : kernel_size,
                    Strides = strides == null ? (1, 1) : strides,
                    Padding = output_padding,
                    DataFormat = data_format,
                    DilationRate = dilation_rate == null ? (1, 1) : dilation_rate,
                    UseBias = use_bias,
                    KernelInitializer = GetInitializerByName(kernel_initializer),
                    BiasInitializer = GetInitializerByName(bias_initializer),
                    Activation = GetActivationByName(activation)
                });

        /// <summary>
        /// Just your regular densely-connected NN layer.
        /// 
        /// Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the 
        /// element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, 
        /// and bias is a bias vector created by the layer (only applicable if use_bias is True).
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix.</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="bias_initializer">Initializer for the bias vector.</param>
        /// <param name="input_shape">N-D tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).</param>
        /// <returns>N-D tensor with shape: (batch_size, ..., units). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).</returns>
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

        /// <summary>
        /// Just your regular densely-connected NN layer.
        /// 
        /// Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the 
        /// element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, 
        /// and bias is a bias vector created by the layer (only applicable if use_bias is True).
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <returns>N-D tensor with shape: (batch_size, ..., units). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).</returns>
        public Dense Dense(int units)
            => new Dense(new DenseArgs
            {
                Units = units,
                Activation = GetActivationByName("linear")
            });

        /// <summary>
        /// Just your regular densely-connected NN layer.
        /// 
        /// Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the 
        /// element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, 
        /// and bias is a bias vector created by the layer (only applicable if use_bias is True).
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="input_shape">N-D tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).</param>
        /// <returns>N-D tensor with shape: (batch_size, ..., units). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).</returns>
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

        /// <summary>
        /// Applies Dropout to the input.
        /// The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, 
        /// which helps prevent overfitting.Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
        /// </summary>
        /// <param name="rate">Float between 0 and 1. Fraction of the input units to drop.</param>
        /// <param name="noise_shape">1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. For instance, 
        /// if your inputs have shape (batch_size, timesteps, features) and you want the dropout mask to be the same for all timesteps, 
        /// you can use noise_shape=(batch_size, 1, features).
        /// </param>
        /// <param name="seed">An integer to use as random seed.</param>
        /// <returns></returns>
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

        /// <summary>
        /// Flattens the input. Does not affect the batch size.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        /// channels_last corresponds to inputs with shape (batch, ..., channels) while channels_first corresponds to inputs with shape (batch, channels, ...). 
        /// It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. 
        /// If you never set it, then it will be "channels_last".
        /// </param>
        /// <returns></returns>
        public Flatten Flatten(string data_format = null)
            => new Flatten(new FlattenArgs
            {
                DataFormat = data_format
            });

        /// <summary>
        /// `Input()` is used to instantiate a Keras tensor.
        ///  Keras tensor is a TensorFlow symbolic tensor object, which we augment with certain attributes that allow us 
        ///  to build a Keras model just by knowing the inputs and outputs of the model.
        /// </summary>
        /// <param name="shape">A shape tuple not including the batch size.</param>
        /// <param name="name">An optional name string for the layer. Should be unique in a model (do not reuse the same name twice). It will be autogenerated if it isn't provided.</param>
        /// <param name="sparse">A boolean specifying whether the placeholder to be created is sparse. Only one of 'ragged' and 'sparse' can be True. 
        /// Note that, if sparse is False, sparse tensors can still be passed into the input - they will be densified with a default value of 0.
        /// </param>
        /// <param name="ragged">A boolean specifying whether the placeholder to be created is ragged. Only one of 'ragged' and 'sparse' can be True. 
        /// In this case, values of 'None' in the 'shape' argument represent ragged dimensions. For more information about RaggedTensors, see this guide.
        /// </param>
        /// <returns>A tensor.</returns>
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

        /// <summary>
        /// Max pooling operation for 1D temporal data.
        /// </summary>
        /// <param name="pool_size">Integer, size of the max pooling window.</param>
        /// <param name="strides">Integer, or null. Specifies how much the pooling window moves for each pooling step. If null, it will default to pool_size.</param>
        /// <param name="padding">One of "valid" or "same" (case-insensitive). "valid" means no padding. 
        /// "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
        /// </param>
        /// <param name="data_format">
        /// A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        /// channels_last corresponds to inputs with shape (batch, steps, features) while channels_first corresponds to inputs with shape (batch, features, steps).
        /// </param>
        /// <returns></returns>
        public MaxPooling1D MaxPooling1D(int? pool_size = null,
            int? strides = null,
            string padding = "valid",
            string data_format = null)
            => new MaxPooling1D(new Pooling1DArgs
            {
                PoolSize = pool_size ?? 2,
                Strides = strides ?? (pool_size ?? 2),
                Padding = padding,
                DataFormat = data_format
            });

        /// <summary>
        /// Max pooling operation for 2D spatial data.
        /// Downsamples the input representation by taking the maximum value over the window defined by pool_size for each dimension along the features axis.
        /// The window is shifted by strides in each dimension. The resulting output when using "valid" padding option has a shape(number of rows or columns) 
        /// of: output_shape = (input_shape - pool_size + 1) / strides)
        /// The resulting output shape when using the "same" padding option is: output_shape = input_shape / strides
        /// </summary>
        /// <param name="pool_size">
        /// Integer or tuple of 2 integers, window size over which to take the maximum. 
        /// (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.
        /// </param>
        /// <param name="strides">
        /// Integer, tuple of 2 integers, or null. Strides values. Specifies how far the pooling window moves for each pooling step. 
        /// If null, it will default to pool_size.
        /// </param>
        /// <param name="padding">One of "valid" or "same" (case-insensitive). "valid" means no padding. 
        /// "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
        /// </param>
        /// <param name="data_format">
        /// A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        /// channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to 
        /// inputs with shape (batch, channels, height, width). 
        /// It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. 
        /// If you never set it, then it will be "channels_last"</param>
        /// <returns></returns>
        public MaxPooling2D MaxPooling2D(TensorShape pool_size = null,
            TensorShape strides = null,
            string padding = "valid",
            string data_format = null)
            => new MaxPooling2D(new MaxPooling2DArgs
            {
                PoolSize = pool_size ?? (2, 2),
                Strides = strides,
                Padding = padding,
                DataFormat = data_format
            });

        /// <summary>
        /// Max pooling layer for 2D inputs (e.g. images).
        /// </summary>
        /// <param name="inputs">The tensor over which to pool. Must have rank 4.</param>
        /// <param name="pool_size">
        /// Integer or tuple of 2 integers, window size over which to take the maximum. 
        /// (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.
        /// </param>
        /// <param name="strides">
        /// Integer, tuple of 2 integers, or null. Strides values. Specifies how far the pooling window moves for each pooling step. 
        /// If null, it will default to pool_size.
        /// </param>
        /// <param name="padding">One of "valid" or "same" (case-insensitive). "valid" means no padding. 
        /// "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
        /// </param>
        /// <param name="data_format">
        /// A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        /// channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to 
        /// inputs with shape (batch, channels, height, width). 
        /// It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. 
        /// If you never set it, then it will be "channels_last"</param>        
        /// <param name="name">A name for the layer</param>
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

        /// <summary>
        /// Fully-connected RNN where the output is to be fed back to input.
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <returns></returns>
        public Layer SimpleRNN(int units) => SimpleRNN(units, "tanh");

        /// <summary>
        /// Fully-connected RNN where the output is to be fed back to input.
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use. If you pass null, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <returns></returns>
        public Layer SimpleRNN(int units,
            Activation activation = null)
                => new SimpleRNN(new SimpleRNNArgs
                {
                    Units = units,
                    Activation = activation
                });

        /// <summary>
        /// 
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="activation">The name of the activation function to use. Default: hyperbolic tangent (tanh)..</param>
        /// <returns></returns>
        public Layer SimpleRNN(int units,
            string activation = "tanh")
                => new SimpleRNN(new SimpleRNNArgs
                {
                    Units = units,
                    Activation = GetActivationByName(activation)
                });

        /// <summary>
        /// Long Short-Term Memory layer - Hochreiter 1997.
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use. If you pass null, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="recurrent_activation">Activation function to use for the recurrent step. If you pass null, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias">Boolean (default True), whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix, used for the linear transformation of the inputs. Default: glorot_uniform.</param>
        /// <param name="recurrent_initializer">Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. Default: orthogonal.</param>
        /// <param name="bias_initializer">Initializer for the bias vector. Default: zeros.</param>
        /// <param name="unit_forget_bias">Boolean (default True). If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force bias_initializer="zeros". This is recommended in Jozefowicz et al..</param>
        /// <param name="dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default: 0.</param>
        /// <param name="recurrent_dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0.</param>
        /// <param name="implementation"></param>
        /// <param name="return_sequences">Boolean. Whether to return the last output. in the output sequence, or the full sequence. Default: False.</param>
        /// <param name="return_state">Whether to return the last state in addition to the output. Default: False.</param>
        /// <param name="go_backwards">Boolean (default false). If True, process the input sequence backwards and return the reversed sequence.</param>
        /// <param name="stateful">Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.</param>
        /// <param name="time_major">
        /// The shape format of the inputs and outputs tensors. If True, the inputs and outputs will be in shape [timesteps, batch, feature], 
        /// whereas in the False case, it will be [batch, timesteps, feature]. Using time_major = True is a bit more efficient because it avoids transposes at the 
        /// beginning and end of the RNN calculation. However, most TensorFlow data is batch-major, so by default this function accepts input and emits output in batch-major form.</param>
        /// <param name="unroll">
        /// Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, 
        /// although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.
        /// </param>
        /// <returns></returns>
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scale"></param>
        /// <param name="offset"></param>
        /// <param name="input_shape"></param>
        /// <returns></returns>
        public Rescaling Rescaling(float scale,
            float offset = 0,
            TensorShape input_shape = null)
            => new Rescaling(new RescalingArgs
            {
                Scale = scale,
                Offset = offset,
                InputShape = input_shape
            });

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public Add Add()
            => new Add(new MergeArgs { });

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public Subtract Subtract()
            => new Subtract(new MergeArgs { });

        /// <summary>
        /// Global max pooling operation for spatial data.
        /// </summary>
        /// <returns></returns>
        public GlobalAveragePooling2D GlobalAveragePooling2D()
            => new GlobalAveragePooling2D(new Pooling2DArgs { });

        /// <summary>
        /// Global average pooling operation for temporal data.
        /// </summary>
        /// <param name="data_format"> A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        /// channels_last corresponds to inputs with shape (batch, steps, features) while channels_first corresponds to inputs with shape (batch, features, steps).
        /// </param>
        /// <returns></returns>
        public GlobalAveragePooling1D GlobalAveragePooling1D(string data_format = "channels_last")
            => new GlobalAveragePooling1D(new Pooling1DArgs { DataFormat = data_format });

        /// <summary>
        /// Global max pooling operation for spatial data.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        /// channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width).</param>
        /// <returns></returns>
        public GlobalAveragePooling2D GlobalAveragePooling2D(string data_format = "channels_last")
            => new GlobalAveragePooling2D(new Pooling2DArgs { DataFormat = data_format });

        /// <summary>
        /// Global max pooling operation for 1D temporal data.
        /// Downsamples the input representation by taking the maximum value over the time dimension.
        /// </summary>
        /// <param name="data_format"> A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        /// channels_last corresponds to inputs with shape (batch, steps, features) while channels_first corresponds to inputs with shape (batch, features, steps).
        /// </param>
        /// <returns></returns>
        public GlobalMaxPooling1D GlobalMaxPooling1D(string data_format = "channels_last")
            => new GlobalMaxPooling1D(new Pooling1DArgs { DataFormat = data_format });

        /// <summary>
        /// Global max pooling operation for spatial data.
        /// </summary>
        /// <param name="data_format">A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        /// channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width).</param>
        /// <returns></returns>
        public GlobalMaxPooling2D GlobalMaxPooling2D(string data_format = "channels_last")
            => new GlobalMaxPooling2D(new Pooling2DArgs { DataFormat = data_format });


        /// <summary>
        /// Get an activation function layer from its name.
        /// </summary>
        /// <param name="name">The name of the activation function. One of linear, relu, sigmoid, and tanh.</param>
        /// <returns></returns>

        Activation GetActivationByName(string name)
            => name switch
            {
                "linear" => keras.activations.Linear,
                "relu" => keras.activations.Relu,
                "sigmoid" => keras.activations.Sigmoid,
                "tanh" => keras.activations.Tanh,
                _ => keras.activations.Linear
            };

        /// <summary>
        /// Get an weights initializer from its name.
        /// </summary>
        /// <param name="name">The name of the initializer. One of zeros, ones, and glorot_uniform.</param>
        /// <returns></returns>
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
