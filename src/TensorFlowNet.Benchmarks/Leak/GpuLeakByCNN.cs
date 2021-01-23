using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Layers;
using NumSharp;
using Tensorflow.Keras;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using BenchmarkDotNet.Attributes;

namespace Tensorflow.Benchmark.Leak
{
    public class GpuLeakByCNN
    {
        protected static LayersApi layers = new LayersApi();
        [Benchmark]
        public void Run()
        {
            // tf.debugging.set_log_device_placement(true);
            tf.Context.Config.GpuOptions.AllowGrowth = true;

            int num = 50, width = 64, height = 64;
            // if width = 128, height = 128, the exception occurs faster

            var bytes = new byte[num * width * height * 3];
            var inputImages = np.array(bytes) / 255.0f;
            inputImages = inputImages.reshape(num, height, width, 3);

            bytes = new byte[num];
            var outLables = np.array(bytes);
            Console.WriteLine("Image.Shape={0}", inputImages.Shape);
            Console.WriteLine("Label.Shape={0}", outLables.Shape);

            tf.enable_eager_execution();

            var inputs = keras.Input((height, width, 3));

            var layer = layers.Conv2D(32, (3, 3), activation: keras.activations.Relu).Apply(inputs);
            layer = layers.MaxPooling2D((2, 2)).Apply(layer);

            layer = layers.Flatten().Apply(layer);

            var outputs = layers.Dense(10).Apply(layer);

            var model = keras.Model(inputs, outputs, "gpuleak");

            model.summary();

            model.compile(loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
             optimizer: keras.optimizers.RMSprop(),
             metrics: new[] { "accuracy" });

            model.fit(inputImages, outLables, batch_size: 32, epochs: 200);

            keras.backend.clear_session();
        }
    }
}
