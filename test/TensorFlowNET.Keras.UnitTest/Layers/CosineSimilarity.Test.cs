using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class CosineSimilarity
    {
        //https://keras.io/api/losses/regression_losses/

        NDArray y_true_float = new float[,] { { 0.0f, 1.0f }, { 1.0f, 1.0f } };
        NDArray y_pred_float = new float[,] { { 1.0f, 0.0f }, { 1.0f, 1.0f } };

        [TestMethod]
      
        public void _Default()
        {
            //>>> # Using 'auto'/'sum_over_batch_size' reduction type.  
            //>>> cosine_loss = tf.keras.losses.CosineSimilarity(axis = 1)
            //>>> # l2_norm(y_true) = [[0., 1.], [1./1.414], 1./1.414]]]  
            //>>> # l2_norm(y_pred) = [[1., 0.], [1./1.414], 1./1.414]]]  
            //>>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]  
            //>>> # loss = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))  
            //>>> #       = -((0. + 0.) +  (0.5 + 0.5)) / 2  
            //-0.5
            var loss = keras.losses.CosineSimilarity(axis : 1);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)(-0.49999997f), call.numpy());
        }

        [TestMethod]

        public void _Sample_Weight()
        {
            //>>> # Calling with 'sample_weight'.  
            //>>> cosine_loss(y_true, y_pred, sample_weight =[0.8, 0.2]).numpy()
            //- 0.0999
            var loss = keras.losses.CosineSimilarity();
            var call = loss.Call(y_true_float, y_pred_float, sample_weight: (NDArray)new float[] { 0.8f, 0.2f });
            Assert.AreEqual((NDArray) (- 0.099999994f), call.numpy());
        }

        [TestMethod]

        public void _SUM()
        {
            //>>> # Using 'sum' reduction type.  
            //>>> cosine_loss = tf.keras.losses.CosineSimilarity(axis = 1,
            //...     reduction = tf.keras.losses.Reduction.SUM)
            //>>> cosine_loss(y_true, y_pred).numpy()
            //- 0.999
            var loss = keras.losses.CosineSimilarity(axis: 1,reduction : ReductionV2.SUM);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)(-0.99999994f), call.numpy());
        }

        [TestMethod]

        public void _None()
        {
            //>>> # Using 'none' reduction type.  
            //>>> cosine_loss = tf.keras.losses.CosineSimilarity(axis = 1,
            //...     reduction = tf.keras.losses.Reduction.NONE)
            //>>> cosine_loss(y_true, y_pred).numpy()
            //array([-0., -0.999], dtype = float32)
            var loss = keras.losses.CosineSimilarity(axis :1, reduction: ReductionV2.NONE);
            var call = loss.Call(y_true_float, y_pred_float);
            Assert.AreEqual((NDArray)new float[] { -0f, -0.99999994f }, call.numpy());
        }

    }
}
