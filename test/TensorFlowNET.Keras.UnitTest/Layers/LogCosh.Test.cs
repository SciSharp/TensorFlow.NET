using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class LogCosh
    {
        //https://keras.io/api/losses/regression_losses/#meansquarederror-class

        NDArray y_true_float = new float[,] { { 0.0f, 1.0f }, { 0.0f, 0.0f } };
        NDArray y_pred_float = new float[,] { { 1.0f, 1.0f }, { 0.0f, 0.0f } };

        [TestMethod]
      
        public void _Default()
        {
            //>>> # Using 'auto'/'sum_over_batch_size' reduction type.  
            //>>> l = tf.keras.losses.LogCosh()
            //>>> l(y_true, y_pred).numpy()
            //0.108
            var loss = keras.losses.LogCosh();
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)0.1084452f, call.numpy());
        }

        [TestMethod]

        public void _Sample_Weight()
        {
        //>>> # Calling with 'sample_weight'.  
        //>>> l(y_true, y_pred, sample_weight =[0.8, 0.2]).numpy()
        //0.087
            var loss = keras.losses.LogCosh();
            var call = loss.Call(y_true_float, y_pred_float, sample_weight: (NDArray)new float[] { 0.8f, 0.2f });
            Assert.AreEqual((NDArray)0.08675616f, call.numpy());
        }

        [TestMethod]

        public void _SUM()
        {
            //>>> # Using 'sum' reduction type.  
            //>>> l = tf.keras.losses.LogCosh(
            //...     reduction = tf.keras.losses.Reduction.SUM)
            //>>> l(y_true, y_pred).numpy()
            //0.217
            var loss = keras.losses.LogCosh(reduction : ReductionV2.SUM);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)0.2168904f, call.numpy());
        }

        [TestMethod]

        public void _None()
        {
            //>>> # Using 'none' reduction type.  
            //>>> l = tf.keras.losses.LogCosh(
            //...     reduction = tf.keras.losses.Reduction.NONE)
            //>>> l(y_true, y_pred).numpy()
            //array([0.217, 0.], dtype = float32)
            var loss = keras.losses.LogCosh(reduction: ReductionV2.NONE);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)new float[] { 0.2168904f, 0.0f }, call.numpy());
        }

    }
}
