using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.NumPy
{
    /// <summary>
    /// https://numpy.org/doc/stable/user/basics.indexing.html
    /// </summary>
    [TestClass]
    public class ArrayIndexingTest : EagerModeTestBase
    {
        [TestMethod]
        public void int_params()
        {
            var x = np.arange(24).reshape((2, 3, 4));
            x[1, 2, 3] = 1;
            var y = x[1, 2, 3];
            Assert.AreEqual(y.shape, Shape.Scalar);
            Assert.AreEqual(y, 1);

            x[0, 0] = new[] { 3, 1, 1, 2 };
            y = x[0, 0];
            Assert.AreEqual(y.shape, 4);
            Assert.AreEqual(y, new[] { 3, 1, 1, 2 });
            
            y = x[0];
            Assert.AreEqual(y.shape, (3, 4));

            var z = np.arange(12).reshape((3, 4));
            x[1] = z;
            Assert.AreEqual(x[1], z);
        }

        [TestMethod]
        public void slice_params()
        {
            var x = np.arange(12).reshape((3, 4));
            var y = x[new Slice(0, 1), new Slice(2)];
            Assert.AreEqual(y.shape, (1, 2));
            Assert.AreEqual(y, np.array(new[] { 2, 3 }).reshape((1, 2)));
        }

        [TestMethod]
        public void slice_string_params()
        {
            var x = np.arange(12).reshape((3, 4));
            var y = x[Slice.ParseSlices("0:1,2:")];
            Assert.AreEqual(y.shape, (1, 2));
            Assert.AreEqual(y, np.array(new[] { 2, 3 }).reshape((1, 2)));
        }

        [TestMethod]
        public void slice_out_bound()
        {
            var input_shape = tf.constant(new int[] { 1, 1 });
            var input_shape_val = input_shape.numpy();
            input_shape_val[(int)input_shape.size - 1] = 1;
            input_shape.Dispose();
        }
    }
}
