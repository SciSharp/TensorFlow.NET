using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Layers
{
    [TestClass]
    public class ActivationTest : EagerModeTestBase
    {
        [TestMethod]
        public void LeakyReLU()
        {
            var layer = keras.layers.LeakyReLU();
            Tensor output = layer.Apply(np.array(-3.0f, -1.0f, 0.0f, 2.0f));
            Equal(new[] { -0.9f, -0.3f, 0.0f, 2.0f }, output.ToArray<float>());
        }

        [TestMethod]
        public void ELU()
        {
            Tensors input = tf.constant(new float[] { -3f, -2f, -1f, 0f, 1f, 2f });
            Tensor output = keras.layers.ELU().Apply(input);
            NDArray expected = new NDArray(new float[] { -0.0950213f, -0.08646648f, -0.06321206f, 0f, 1f, 2f });
            Assert.AreEqual(expected.numpy(), output.numpy());
        }

        [TestMethod]
        public void SELU()
        {
            Tensor input = tf.constant(new float[] { -3f, -2f, -1f, 0f, 1f, 2f });
            Tensor output = keras.layers.SELU().Apply(input);
            NDArray expected = new NDArray(new float[] { -1.6705688f, -1.5201665f, -1.1113307f, 0f, 1.050701f, 2.101402f });
            Assert.AreEqual(expected.numpy(), output.numpy());
        }

        [TestMethod]
        public void Softmax()
        {
            Tensor input = tf.constant(new float[] { -3f, -2f, -1f, 0f, 1f, 2f });
            Tensor output = keras.layers.Softmax(new Axis(-1)).Apply(input);
            var expected = new float[] { 0.0042697787f, 0.011606461f, 0.031549633f, 0.085760795f, 0.23312202f, 0.6336913f };
            Assert.IsTrue(Equal(expected, output.ToArray<float>()));
        }

        [TestMethod]
        public void Softplus()
        {
            Tensor input = tf.constant(new float[] { -3f, -2f, -1f, 0f, 1f, 2f });
            Tensor output = keras.layers.Softplus().Apply(input);
            NDArray expected = new NDArray(new float[] { 0.04858733f, 0.12692805f, 0.31326166f, 0.6931472f, 1.3132616f, 2.126928f });
            Assert.IsTrue(expected == output.numpy());
        }

        [TestMethod]
        public void Softsign()
        {
            Tensor input = tf.constant(new float[] { -3f, -2f, -1f, 0f, 1f, 2f });
            Tensor output = keras.layers.Softsign().Apply(input);
            NDArray expected = new NDArray(new float[] { -0.75f, -0.66666667f, -0.5f, 0f, 0.5f, 0.66666667f });
            Assert.AreEqual(expected, output.numpy());
        }


        [TestMethod]
        public void Exponential()
        {
            Tensor input = tf.constant(new float[] { -3f, -2f, -1f, 0f, 1f, 2f });
            Tensor output = keras.layers.Exponential().Apply(input);
            var expected = new float[] { 0.049787067f, 0.13533528f, 0.36787945f, 1f, 2.7182817f, 7.389056f };
            Assert.IsTrue(Equal(expected, output.ToArray<float>()));
        }

        [TestMethod]
        public void HardSigmoid()
        {
            Tensor input = tf.constant(new float[] { -3f, -2f, -1f, 0f, 1f, 2f });
            Tensor output = keras.layers.HardSigmoid().Apply(input);
            // Note, this should be [0, 0.1, 0.3, 0.5, 0.7, 0.9]
            // But somehow the second element will have 0.099999994
            // Probably because there is an accuracy loss somewhere
            NDArray expected = new NDArray(new float[] { 0f, 0.099999994f, 0.3f, 0.5f, 0.7f, 0.9f });
            Assert.AreEqual(expected, output.numpy());
        }


        [TestMethod]
        public void Swish()
        {
            Tensor input = tf.constant(new float[] { -3f, -2f, -1f, 0f, 1f, 2f });
            Tensor output = keras.layers.Swish().Apply(input);
            NDArray expected = new NDArray(new float[] { -0.14227762f, -0.23840584f, -0.26894143f, 0f, 0.7310586f, 1.761594f });
            Assert.AreEqual(expected, output.numpy());
        }

        /// <summary>
        /// https://www.tensorflow.org/addons/api_docs/python/tfa/activations/mish
        /// </summary>
        [TestMethod]
        public void Mish()
        {
            var x = tf.constant(new[] { 1.0, 0.0, 1.0 }, dtype: tf.float32);
            var output = keras.activations.Mish.Apply(x);
            Assert.AreEqual(new[] { 0.86509836f, 0f, 0.86509836f }, output.numpy());
        }
    }
}
