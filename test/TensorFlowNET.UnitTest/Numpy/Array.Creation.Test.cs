using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.NumPy;

namespace TensorFlowNET.UnitTest.NumPy
{
    /// <summary>
    /// https://numpy.org/doc/stable/reference/routines.array-creation.html
    /// </summary>
    [TestClass]
    public class ArrayCreationTest : EagerModeTestBase
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
            AssetSequenceEqual(new[] { 0, 1, 2 }, x.ToArray<int>());

            x = np.arange(3f);
            Assert.IsTrue(Equal(new float[] { 0, 1, 2 }, x.ToArray<float>()));

            var y = np.arange(3, 7);
            AssetSequenceEqual(new[] { 3, 4, 5, 6 }, y.ToArray<int>());

            y = np.arange(3, 7, 2);
            AssetSequenceEqual(new[] { 3, 5 }, y.ToArray<int>());
        }

        [TestMethod]
        public void array()
        {
            var x = np.array(1, 2, 3);
            AssetSequenceEqual(new[] { 1, 2, 3 }, x.ToArray<int>());

            x = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            AssetSequenceEqual(new[] { 1, 2, 3, 4, 5, 6 }, x.ToArray<int>());
        }

        [TestMethod]
        public void eye()
        {
            var x = np.eye(3, k: 1);
            Assert.IsTrue(Equal(new double[] { 0, 1, 0, 0, 0, 1, 0, 0, 0 }, x.ToArray<double>()));
        }

        [TestMethod]
        public void linspace()
        {
            var x = np.linspace(2.0, 3.0, num: 5);
            Assert.IsTrue(Equal(new double[] { 2, 2.25, 2.5, 2.75, 3 }, x.ToArray<double>()));

            x = np.linspace(2.0, 3.0, num: 5, endpoint: false);
            Assert.IsTrue(Equal(new double[] { 2, 2.2, 2.4, 2.6, 2.8 }, x.ToArray<double>()));
        }

        [TestMethod]
        public void meshgrid()
        {
            var x = np.linspace(0, 1, num: 3);
            var y = np.linspace(0, 1, num: 2);
            var (xv, yv) = np.meshgrid(x, y);
            Assert.IsTrue(Equal(new double[] { 0, 0.5, 1, 0, 0.5, 1 }, xv.ToArray<double>()));
            Assert.IsTrue(Equal(new double[] { 0, 0, 0, 1, 1, 1 }, yv.ToArray<double>()));

            (xv, yv) = np.meshgrid(x, y, sparse: true);
            Assert.IsTrue(Equal(new double[] { 0, 0.5, 1 }, xv.ToArray<double>()));
            AssetSequenceEqual(new long[] { 1, 3 }, xv.shape.dims);
            Assert.IsTrue(Equal(new double[] { 0, 1 }, yv.ToArray<double>()));
            AssetSequenceEqual(new long[] { 2, 1 }, yv.shape.dims);
        }

        [TestMethod]
        public void meshgrid_same_ndim()
        {
            var (a, b) = np.meshgrid(np.arange(3), np.arange(3));
            AssetSequenceEqual(a.ToArray<int>(), new int[] { 0, 1, 2, 0, 1, 2, 0, 1, 2 });
            AssetSequenceEqual(b.ToArray<int>(), new int[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 });
        }
    }
}
