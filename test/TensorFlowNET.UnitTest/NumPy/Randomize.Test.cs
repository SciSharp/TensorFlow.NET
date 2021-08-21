using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.NumPy;

namespace TensorFlowNET.UnitTest.NumPy
{
    /// <summary>
    /// https://numpy.org/doc/1.20/reference/random/index.html
    /// </summary>
    [TestClass]
    public class RandomizeTest : EagerModeTestBase
    {
        [TestMethod]
        public void permutation()
        {
            var x = np.random.permutation(10);
            Assert.AreEqual(x.shape, 10);
            var y = np.random.permutation(x);
            Assert.AreEqual(x.shape, 10);
            Assert.AreNotEqual(x.ToArray<int>(), y.ToArray<int>());
        }

        /// <summary>
        /// https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        /// </summary>
        [TestMethod]
        public void normal()
        {
            var x = np.random.normal(0, 0.1f, 1000);
            Equal(np.mean(x), 0f);
        }

        [TestMethod]
        public void randn()
        {
            var x = np.random.randn();
            Assert.AreEqual(np.float32, x.dtype);

            x = np.random.randn(2, 4);
            Equal(np.mean(x), 0f);
        }
    }
}
