using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Tensorflow.Binding;

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

            // one dimensions input with unknown batchsize
            var input_2 = tf.keras.layers.Input((60));
            var dense_2 = tf.keras.layers.Dense(64);
            var output_2 = dense.Apply(input_2);
            var model_2 = tf.keras.Model(input_2, output_2);

            // two dimensions input with specified batchsize
            var input_3 = tf.keras.layers.Input((17, 60), 8);
            var dense_3 = tf.keras.layers.Dense(64);
            var output_3 = dense.Apply(input_3);
            var model_3 = tf.keras.Model(input_3, output_3);

            // one dimensions input with specified batchsize
            var input_4 = tf.keras.layers.Input((60), 8);
            var dense_4 = tf.keras.layers.Dense(64);
            var output_4 = dense.Apply(input_4);
            var model_4 = tf.keras.Model(input_4, output_4);
        }
    }
}
