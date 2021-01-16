using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class Huber
    {
        //https://keras.io/api/losses/regression_losses/#meansquarederror-class

        NDArray y_true_float = new float[,] { { 0.0f, 1.0f }, { 0.0f, 0.0f } };
        NDArray y_pred_float = new float[,] { { 0.6f, 0.4f }, { 0.4f, 0.6f } };

        [TestMethod]
      
        public void _Default()
        {
            //>>> # Using 'auto'/'sum_over_batch_size' reduction type.  
            //>>> h = tf.keras.losses.Huber()
            //>>> h(y_true, y_pred).numpy()
            //0.155
            var loss = keras.losses.Huber();
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)0.155f, call.numpy());
        }

        [TestMethod]

        public void _Sample_Weight()
        {
            //>>> # Calling with 'sample_weight'.  
            //>>> h(y_true, y_pred, sample_weight =[1, 0]).numpy()
            //0.09
            var loss = keras.losses.Huber();
            var call = loss.Call(y_true_float, y_pred_float, sample_weight: (NDArray)new float[] { 0.1f, 0.0f });
            Assert.AreEqual((NDArray)0.009000001f, call.numpy());
        }

        [TestMethod]

        public void _SUM()
        {
            //>>> # Using 'sum' reduction type.  
            //>>> h = tf.keras.losses.Huber(
            //...     reduction = tf.keras.losses.Reduction.SUM)
            //>>> h(y_true, y_pred).numpy()
            //0.31
            var loss = keras.losses.Huber(reduction : ReductionV2.SUM);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)0.31f, call.numpy());
        }

        [TestMethod]

        public void _None()
        {
            //>>> # Using 'none' reduction type.  
            //>>> h = tf.keras.losses.Huber(
            //...     reduction = tf.keras.losses.Reduction.NONE)
            //>>> h(y_true, y_pred).numpy()
            //array([0.18, 0.13], dtype = float32)
            var loss = keras.losses.Huber(reduction: ReductionV2.NONE);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)new float[] { 0.18f, 0.13000001f }, call.numpy());
        }

    }
}
