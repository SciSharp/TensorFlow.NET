using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.functional_ops_test
{
    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/scan
    /// </summary>
    [TestClass]
    public class ScanTestCase
    {
        [TestMethod]
        public void ScanForward()
        {
            var fn = new Func<Tensor, Tensor, Tensor>((a, x) => tf.add(a, x));
            
            var sess = tf.Session().as_default();

            var input = tf.placeholder(TF_DataType.TF_INT32, new TensorShape(6));
            var scan = functional_ops.scan(fn, input);
            sess.run(scan, (input, np.array(1,2,3,4,5,6))).Should().Be(np.array(1,3,6,10,15,21));
        }

        [TestMethod]
        public void ScanReverse()
        {
            var fn = new Func<Tensor, Tensor, Tensor>((a, x) => tf.add(a, x));
            
            var sess = tf.Session().as_default();

            var input = tf.placeholder(TF_DataType.TF_INT32, new TensorShape(6));
            var scan = functional_ops.scan(fn, input, reverse:true);
            sess.run(scan, (input, np.array(1,2,3,4,5,6))).Should().Be(np.array(21,20,18,15,11,6));
        }
    }
}
