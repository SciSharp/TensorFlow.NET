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
    /// https://numpy.org/doc/stable/user/basics.indexing.html
    /// </summary>
    [TestClass]
    public class ArraySortingTest : EagerModeTestBase
    {
        /// <summary>
        /// https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
        /// </summary>
        [TestMethod]
        public void argsort()
        {
            var x = np.array(new[] { 3, 1, 2 });
            var ind = np.argsort(x);
            Assert.AreEqual(ind, new[] { 1, 2, 0 });

            var y = np.array(new[,] { { 0, 3 }, { 2, 2 } });
            ind = np.argsort(y, axis: 0);
            Assert.AreEqual(ind[0], new[] { 0, 1 });
            Assert.AreEqual(ind[1], new[] { 1, 0 });
        }

        /// <summary>
        /// https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        /// </summary>
        [TestMethod]
        public void sort()
        {
            var x = np.array(new int[] { 3, 1, 2 });
            var sorted = np.sort(x);
            // Assert.IsTrue(sorted.ToArray<int>() is [1, 2, 3]);
        }
    }
}
