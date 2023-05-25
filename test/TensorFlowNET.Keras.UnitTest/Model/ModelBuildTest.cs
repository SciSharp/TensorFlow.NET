using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Model
{
    [TestClass]
    public class ModelBuildTest
    {
        [TestMethod]
        public void DenseBuild()
        {
            // two dimensions input with unknown batchsize
            var input = tf.keras.layers.Input((17, 60));
            var dense = tf.keras.layers.Dense(64);
            var output = dense.Apply(input);
            var model = tf.keras.Model(input, output);
            model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.CategoricalCrossentropy());

            // one dimensions input with unknown batchsize
            var input_2 = tf.keras.layers.Input((60));
            var dense_2 = tf.keras.layers.Dense(64);
            var output_2 = dense_2.Apply(input_2);
            var model_2 = tf.keras.Model(input_2, output_2);
            model_2.compile(tf.keras.optimizers.Adam(), tf.keras.losses.CategoricalCrossentropy());

            // two dimensions input with specified batchsize
            var input_3 = tf.keras.layers.Input((17, 60), 8);
            var dense_3 = tf.keras.layers.Dense(64);
            var output_3 = dense_3.Apply(input_3);
            var model_3 = tf.keras.Model(input_3, output_3);
            model_3.compile(tf.keras.optimizers.Adam(), tf.keras.losses.CategoricalCrossentropy());

            // one dimensions input with specified batchsize
            var input_4 = tf.keras.layers.Input((60), 8);
            var dense_4 = tf.keras.layers.Dense(64);
            var output_4 = dense_4.Apply(input_4);
            var model_4 = tf.keras.Model(input_4, output_4);
            model_4.compile(tf.keras.optimizers.Adam(), tf.keras.losses.CategoricalCrossentropy());
        }

        [TestMethod]
        public void NestedSequential()
        {
            var block1 = keras.Sequential(new[] { 
                keras.layers.InputLayer((3, 3)), 
                keras.Sequential(new []
                    {
                        keras.layers.Flatten(),
                        keras.layers.Dense(5)
                    }
                )
            });
            block1.compile(tf.keras.optimizers.Adam(), tf.keras.losses.CategoricalCrossentropy());

            var x = tf.ones((1, 3, 3));
            var y = block1.predict(x);
            Console.WriteLine(y);
        }
    }
}
