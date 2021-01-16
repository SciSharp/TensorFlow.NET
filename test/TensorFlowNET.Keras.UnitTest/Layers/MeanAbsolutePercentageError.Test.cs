using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class MeanAbsolutePercentageError
    {
        //https://keras.io/api/losses/regression_losses/

        NDArray y_true_float = new float[,] { { 2.0f, 1.0f }, { 2.0f, 3.0f } };
        NDArray y_pred_float = new float[,] { { 1.0f, 1.0f }, { 1.0f, 0.0f } };

        [TestMethod]

        public void _Default()
        {
            //>>> # Using 'auto'/'sum_over_batch_size' reduction type.  
            //>>> mape = tf.keras.losses.MeanAbsolutePercentageError()
            //>>> mape(y_true, y_pred).numpy()
            //50.
            var loss = keras.losses.MeanAbsolutePercentageError();
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)(50f), call.numpy());
        }

        [TestMethod]

        public void _Sample_Weight()
        {
            //>>> # Calling with 'sample_weight'.  
            //>>> mape(y_true, y_pred, sample_weight =[0.7, 0.3]).numpy()
            //20.
            var loss = keras.losses.MeanAbsolutePercentageError();
            var call = loss.Call(y_true_float, y_pred_float, sample_weight: (NDArray)new float[] { 0.7f, 0.3f });
            Assert.AreEqual((NDArray)(20f), call.numpy());
        }

        [TestMethod]

        public void _SUM()
        {
            //>>> # Using 'sum' reduction type.  
            //>>> mape = tf.keras.losses.MeanAbsolutePercentageError(
            //...     reduction = tf.keras.losses.Reduction.SUM)
            //>>> mape(y_true, y_pred).numpy()
            //100.
            var loss = keras.losses.MeanAbsolutePercentageError( reduction: ReductionV2.SUM);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)(100f), call.numpy());
        }

        [TestMethod]

        public void _None()
        {
            //>>> # Using 'none' reduction type.  
            //>>> mape = tf.keras.losses.MeanAbsolutePercentageError(
            //...     reduction = tf.keras.losses.Reduction.NONE)
            //>>> mape(y_true, y_pred).numpy()
            //array([25., 75.], dtype = float32)
            var loss = keras.losses.MeanAbsolutePercentageError(reduction: ReductionV2.NONE);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)new float[] { 25f, 75f }, call.numpy());
        }

    }
}
