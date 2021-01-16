using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class MeanSquaredLogarithmicError
    {
        //https://keras.io/api/losses/regression_losses/

        NDArray y_true_float = new float[,] { { 0.0f, 1.0f }, { 0.0f, 0.0f } };
        NDArray y_pred_float = new float[,] { { 1.0f, 1.0f }, { 1.0f, 0.0f } };

        [TestMethod]

        public void _Default()
        {
            //>>> # Using 'auto'/'sum_over_batch_size' reduction type.  
            //>>> msle = tf.keras.losses.MeanSquaredLogarithmicError()
            //>>> msle(y_true, y_pred).numpy()
            //0.240
            var loss = keras.losses.MeanSquaredLogarithmicError();
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)(0.24022643f), call.numpy());
        }

        [TestMethod]

        public void _Sample_Weight()
        {
            //>>> # Calling with 'sample_weight'.  
            //>>> msle(y_true, y_pred, sample_weight =[0.7, 0.3]).numpy()
            //0.120
            var loss = keras.losses.MeanSquaredLogarithmicError();
            var call = loss.Call(y_true_float, y_pred_float, sample_weight: (NDArray)new float[] { 0.7f, 0.3f });
            Assert.AreEqual((NDArray)(0.12011322f), call.numpy());
        }

        [TestMethod]

        public void _SUM()
        {
            //>>> # Using 'sum' reduction type.  
            //>>> msle = tf.keras.losses.MeanSquaredLogarithmicError(
            //...     reduction = tf.keras.losses.Reduction.SUM)
            //>>> msle(y_true, y_pred).numpy()
            //0.480
            var loss = keras.losses.MeanSquaredLogarithmicError( reduction: ReductionV2.SUM);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)(0.48045287f), call.numpy());
        }

        [TestMethod]

        public void _None()
        {
            //>>> # Using 'none' reduction type.  
            //>>> msle = tf.keras.losses.MeanSquaredLogarithmicError(
            //...     reduction = tf.keras.losses.Reduction.NONE)
            //>>> msle(y_true, y_pred).numpy()
            //array([0.240, 0.240], dtype = float32)
            var loss = keras.losses.MeanSquaredLogarithmicError(reduction: ReductionV2.NONE);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)new float[] { 0.24022643f, 0.24022643f }, call.numpy());
        }

    }
}
