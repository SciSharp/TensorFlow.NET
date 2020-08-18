using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.TF_API
{
    [TestClass]
    public class tfdata_test
    {
        [TestMethod]
        public void CreateFromTensor()
        {
            var X = tf.constant(new[] { 2013, 2014, 2015, 2016, 2017 });
            var Y = tf.constant(new[] { 12000, 14000, 15000, 16500, 17500 });

            var dataset = tf.data.Dataset.from_tensor_slices(X, Y);
            int n = 0;
            foreach (var (item_x, item_y) in dataset)
            {
                print($"x:{item_x.numpy()},y:{item_y.numpy()}");
                n += 1;
            }
            Assert.AreEqual(5, n);
        }
       
    }
}
