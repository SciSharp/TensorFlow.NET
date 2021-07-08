using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Numpy;

namespace TensorFlowNET.UnitTest.Numpy
{
    /// <summary>
    /// https://numpy.org/doc/stable/reference/routines.array-creation.html
    /// </summary>
    [TestClass]
    public class NumpyArrayCreationTest
    {
        [TestMethod]
        public void empty_zeros_ones_full()
        {
            var empty = np.empty((2, 2));
            var zeros = np.zeros((2, 2));
            var ones = np.ones((2, 2));
            var full = np.full((2, 2), 0.1f);
        }

        [TestMethod]
        public void arange()
        {
            var x = np.arange(3);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 0, 1, 2 }, x.Data<int>()));

            x = np.arange(3f);
            Assert.IsTrue(Enumerable.SequenceEqual(new float[] { 0, 1, 2 }, x.Data<float>()));

            var y = np.arange(3, 7);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 3, 4, 5, 6 }, y.Data<int>()));
            y = np.arange(3, 7, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 3, 5 }, y.Data<int>()));
        }

        [TestMethod]
        public void array()
        {
            var x = np.array(1, 2, 3);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 1, 2, 3 }, x.Data<int>()));

            x = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 1, 2, 3, 4, 5, 6 }, x.Data<int>()));
        }
    }
}
