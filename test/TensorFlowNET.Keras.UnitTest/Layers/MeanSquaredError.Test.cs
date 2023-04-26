using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Layers
{
    [TestClass]
    public class MeanSquaredErrorTest
    {
        //https://keras.io/api/losses/regression_losses/#meansquarederror-class

        private NDArray y_true = new double[,] { { 0.0, 1.0 }, { 0.0, 0.0 } };
        private NDArray y_pred = new double[,] { { 1.0, 1.0 }, { 1.0, 0.0 } };

        [TestMethod]

        public void Mse_Double()
        {
            var mse = keras.losses.MeanSquaredError();
            var call = mse.Call(y_true, y_pred);
            Assert.AreEqual(call.numpy(), 0.5);
        }

        [TestMethod]

        public void Mse_Float()
        {
            NDArray y_true_float = new float[,] { { 0.0f, 1.0f }, { 0.0f, 0.0f } };
            NDArray y_pred_float = new float[,] { { 1.0f, 1.0f }, { 1.0f, 0.0f } };

            var mse = keras.losses.MeanSquaredError();
            var call = mse.Call(y_true_float, y_pred_float);
            Assert.AreEqual(call.numpy(), 0.5f);
        }

        [TestMethod]

        public void Mse_Sample_Weight()
        {
            var mse = keras.losses.MeanSquaredError();
            var call = mse.Call(y_true, y_pred, sample_weight: (NDArray)new double[] { 0.7, 0.3 });
            Assert.AreEqual(call.numpy(), 0.25);
        }

        [TestMethod]
        public void Mse_Reduction_SUM()
        {
            var mse = keras.losses.MeanSquaredError(reduction: Reduction.SUM);
            var call = mse.Call(y_true, y_pred);
            Assert.AreEqual(call.numpy(), 1.0);
        }

        [TestMethod]

        public void Mse_Reduction_NONE()
        {
            var mse = keras.losses.MeanSquaredError(reduction: Reduction.NONE);
            var call = mse.Call(y_true, y_pred);
            Assert.AreEqual(call.numpy(), new double[] { 0.5, 0.5 });
        }
    }
}
