using NumSharp;
using System;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine.DataAdapters;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow
{
    class MemoryBasicTest
    {
        public Action<int, int> Placeholder
            => (epoch, iterate) =>
            {
                var ph = array_ops.placeholder(tf.float32, (10, 512, 512, 3));
            };

        /// <summary>
        /// 
        /// </summary>
        public Action<int, int> Constant
            => (epoch, iterate) =>
            {
                var tensor = tf.constant(3112.0f);
            };

        public Action<int, int> Constant2x3
            => (epoch, iterate) =>
            {
                var nd = np.arange(1000).reshape(10, 100);
                var tensor = tf.constant(nd);
                var data = tensor.numpy();
            };

        public Action<int, int> ConstantString
            => (epoch, iterate) =>
            {
                var tensor = tf.constant(new string[] 
                {
                    "Biden immigration bill would put millions of illegal immigrants on 8-year fast-track to citizenship",
                    "The Associated Press, which also reported that the eight-year path is in the bill.",
                    "The bill would also include provisions to stem the flow of migration by addressing root causes of migration from south of the border."
                });
                var data = tensor.numpy();
            };

        public Action<int, int> Variable
            => (epoch, iterate) =>
            {
                var nd = np.arange(1 * 256 * 256 * 3).reshape(1, 256, 256, 3);
                ResourceVariable variable = tf.Variable(nd);
            };

        public Action<int, int> VariableRead
            => (epoch, iterate) =>
            {
                var nd = np.zeros(1 * 256 * 256 * 3).astype(np.float32).reshape(1, 256, 256, 3);
                ResourceVariable variable = tf.Variable(nd);

                for (int i = 0; i< 10; i++)
                {
                    var v = variable.numpy();
                }
            };

        public Action<int, int> VariableAssign
            => (epoch, iterate) =>
            {
                ResourceVariable variable = tf.Variable(3112f);
                AssignVariable(variable);
                for (int i = 0; i < 100; i++)
                {
                    var v = variable.numpy();
                    if ((float)v != 1984f)
                        throw new ValueError("");
                }
            };

        void AssignVariable(IVariableV1 v)
        {
            using var tensor = tf.constant(1984f);
            v.assign(tensor);
        }

        public Action<int, int> MathAdd
            => (epoch, iterate) =>
            {
                var x = tf.constant(3112.0f);
                var y = tf.constant(3112.0f);
                var z = x + y;
            };

        public Action<int, int> Gradient
            => (epoch, iterate) =>
            {
                var w = tf.constant(3112.0f);
                using var tape = tf.GradientTape();
                tape.watch(w);
                var loss = w * w;
                var grad = tape.gradient(loss, w);
            };

        public Action<int, int> Conv2DWithTensor
            => (epoch, iterate) =>
            {
                var input = array_ops.zeros((10, 32, 32, 3), dtypes.float32);
                var filter = array_ops.zeros((3, 3, 3, 32), dtypes.float32);
                var strides = new[] { 1, 1, 1, 1 };
                var dilations = new[] { 1, 1, 1, 1 };

                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Conv2D", null,
                    null,
                    input, filter,
                    "strides", strides,
                    "use_cudnn_on_gpu", true,
                    "padding", "VALID",
                    "explicit_paddings", new int[0],
                    "data_format", "NHWC",
                    "dilations", dilations);
            };

        public Action<int, int> Conv2DWithVariable
            => (epoch, iterate) =>
            {
                var input = array_ops.zeros((10, 32, 32, 3), dtypes.float32);
                var filter = tf.Variable(array_ops.zeros((3, 3, 3, 32), dtypes.float32));
                var strides = new[] { 1, 1, 1, 1 };
                var dilations = new[] { 1, 1, 1, 1 };

                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Conv2D", null,
                    null,
                    input, filter,
                    "strides", strides,
                    "use_cudnn_on_gpu", true,
                    "padding", "VALID",
                    "explicit_paddings", new int[0],
                    "data_format", "NHWC",
                    "dilations", dilations);
            };

        public Action<int, int> Dataset
            => (epoch, iterate) =>
            {
                TensorShape shape = (16, 32, 32, 3);
                var images = np.arange(shape.size).astype(np.float32).reshape(shape.dims);
                var data_handler = new DataHandler(new DataHandlerArgs
                {
                    X = images,
                    BatchSize = 2,
                    StepsPerEpoch = -1,
                    InitialEpoch = 0,
                    Epochs = 2,
                    MaxQueueSize = 10,
                    Workers = 1,
                    UseMultiprocessing = false,
                    StepsPerExecution = tf.Variable(1)
                });

                /*foreach (var (_epoch, iterator) in data_handler.enumerate_epochs())
                {
                    foreach (var step in data_handler.steps())
                        iterator.next();
                }*/
            };
    }
}
