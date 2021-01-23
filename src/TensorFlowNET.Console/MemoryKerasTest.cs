using NumSharp;
using System;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow
{
    class MemoryKerasTest
    {
        public Action<int, int> Conv2DLayer
            => (epoch, iterate) =>
            {
                var input_shape = new int[] { 4, 512, 512, 3 };
                var x = tf.random.normal(input_shape);
                var conv2d = keras.layers.Conv2D(2, 3, activation: keras.activations.Relu);
                var output = conv2d.Apply(x);
            };

        public Action<int, int> InputLayer
            => (epoch, iterate) =>
            {
                TensorShape shape = (32, 256, 256, 3); // 48M
                var images = np.arange(shape.size).astype(np.float32).reshape(shape.dims);

                var inputs = keras.Input((shape.dims[1], shape.dims[2], 3));
                var conv2d = keras.layers.Conv2D(32, kernel_size: (3, 3),
                    activation: keras.activations.Linear);
                var outputs = conv2d.Apply(inputs);
            };

        public Action<int, int> Prediction
            => (epoch, iterate) =>
            {
                TensorShape shape = (32, 256, 256, 3); // 48M
                var images = np.arange(shape.size).astype(np.float32).reshape(shape.dims);

                var inputs = keras.Input((shape.dims[1], shape.dims[2], 3));
                var conv2d = keras.layers.Conv2D(32, kernel_size: (3, 3),
                    activation: keras.activations.Linear).Apply(inputs);

                var flatten = keras.layers.Flatten().Apply(inputs);
                var outputs = keras.layers.Dense(10).Apply(flatten);

                var model = keras.Model(inputs, outputs, "prediction");
                for (int i = 0; i < 10; i++)
                {
                    model.predict(images, batch_size: 8);
                }
            };
    }
}
