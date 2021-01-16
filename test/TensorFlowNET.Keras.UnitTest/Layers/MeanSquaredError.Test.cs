using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
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
            Assert.AreEqual((NDArray)0.5, call.numpy()) ;
        }

        [TestMethod]
      
        public void Mse_Float()
        {
            NDArray y_true_float = new float[,] { { 0.0f, 1.0f }, { 0.0f, 0.0f } };
            NDArray y_pred_float = new float[,] { { 1.0f, 1.0f }, { 1.0f, 0.0f } };

            var mse = keras.losses.MeanSquaredError();
            var call = mse.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)0.5, call.numpy());
        }

        [TestMethod]

        public void Mse_Sample_Weight()
        {
            var mse = keras.losses.MeanSquaredError();
            var call = mse.Call(y_true, y_pred, sample_weight: (NDArray)new double[] { 0.7, 0.3 });
            Assert.AreEqual((NDArray)0.25, call.numpy());
        }

        [TestMethod]
        public void Mse_Reduction_SUM()
        {
            var mse = keras.losses.MeanSquaredError(reduction: Reduction.SUM);
            var call = mse.Call(y_true, y_pred);
            Assert.AreEqual((NDArray)1.0, call.numpy());
        }

        [TestMethod]

        public void Mse_Reduction_NONE()
        {
            var mse = keras.losses.MeanSquaredError(reduction: Reduction.NONE);
            var call = mse.Call(y_true, y_pred);
            Assert.AreEqual((NDArray)new double[] { 0.5, 0.5 }, call.numpy());
        }
    }
}
