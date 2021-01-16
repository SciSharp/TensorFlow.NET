using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class MeanAbsoluteError
    {
        //https://keras.io/api/losses/regression_losses/

        NDArray y_true_float = new float[,] { { 0.0f, 1.0f }, { 0.0f, 0.0f } };
        NDArray y_pred_float = new float[,] { { 1.0f, 1.0f }, { 1.0f, 0.0f } };

        [TestMethod]

        public void _Default()
        {

            //>>> # Using 'auto'/'sum_over_batch_size' reduction type.  
            //>>> mae = tf.keras.losses.MeanAbsoluteError()
            //>>> mae(y_true, y_pred).numpy()
            //0.5
            var loss = keras.losses.MeanAbsoluteError();
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)(0.5f), call.numpy());
        }

        [TestMethod]

        public void _Sample_Weight()
        {
            //>>> # Calling with 'sample_weight'.  
            //>>> mae(y_true, y_pred, sample_weight =[0.7, 0.3]).numpy()
            //0.25
            var loss = keras.losses.MeanAbsoluteError();
            var call = loss.Call(y_true_float, y_pred_float, sample_weight: (NDArray)new float[] { 0.7f, 0.3f });
            Assert.AreEqual((NDArray)(0.25f), call.numpy());
        }

        [TestMethod]

        public void _SUM()
        {
            //>>> # Using 'sum' reduction type.  
            //>>> mae = tf.keras.losses.MeanAbsoluteError(
            //...     reduction = tf.keras.losses.Reduction.SUM)
            //>>> mae(y_true, y_pred).numpy()
            //1.0
            var loss = keras.losses.MeanAbsoluteError( reduction: ReductionV2.SUM);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)(1.0f), call.numpy());
        }

        [TestMethod]

        public void _None()
        {
            //>>> # Using 'none' reduction type.  
            //>>> mae = tf.keras.losses.MeanAbsoluteError(
            //...     reduction = tf.keras.losses.Reduction.NONE)
            //>>> mae(y_true, y_pred).numpy()
            //array([0.5, 0.5], dtype = float32)
            var loss = keras.losses.MeanAbsoluteError(reduction: ReductionV2.NONE);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)new float[] { 0.5f, 0.5f }, call.numpy());
        }

    }
}
