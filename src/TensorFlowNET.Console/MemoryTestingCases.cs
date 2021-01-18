using NumSharp;
using System;
using static Tensorflow.Binding;

namespace Tensorflow
{
    class MemoryTestingCases
    {
        /// <summary>
        /// 
        /// </summary>
        public Action<int> Constant
            => (iterate) =>
            {
                for (int i = 0; i < iterate; i++)
                {
                    var tensor = tf.constant(3112.0f);
                }
            };

        public Action<int> Constant2x3
            => (iterate) =>
            {
                var nd = np.arange(1000).reshape(10, 100);
                for (int i = 0; i < iterate; i++)
                {
                    var tensor = tf.constant(nd);
                    var data = tensor.numpy();
                }
            };

        public Action<int> Variable
            => (iterate) =>
            {
                for (int i = 0; i < iterate; i++)
                {
                    var nd = np.arange(128 * 128 * 3).reshape(128, 128, 3);
                    var variable = tf.Variable(nd);
                }
            };

        public Action<int> MathAdd
            => (iterate) =>
            {
                var x = tf.constant(3112.0f);
                var y = tf.constant(3112.0f);

                for (int i = 0; i < iterate; i++)
                {
                    var z = x + y;
                }
            };

        public Action<int> Gradient
            => (iterate) =>
            {
                for (int i = 0; i < iterate; i++)
                {
                    var w = tf.constant(3112.0f);
                    using var tape = tf.GradientTape();
                    tape.watch(w);
                    var loss = w * w;
                    var grad = tape.gradient(loss, w);
                }
            };

        public Action<int> Conv2dWithVariable
            => (iterate) =>
            {
                for (int i = 0; i < iterate; i++)
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
                }
            };

        public Action<int> Conv2dWithTensor
            => (iterate) =>
            {
                for (int i = 0; i < iterate; i++)
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
                }
            };
    }
}
