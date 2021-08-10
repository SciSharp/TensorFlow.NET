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
            Assert.AreEqual(np.float32, full.dtype);
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
        public void to_multi_dim_array()
        {
            var x1 = np.arange(12);
            var y1 = x1.ToMultiDimArray<int>();
            AssetSequenceEqual((int[])y1, x1.ToArray<int>());

            var x2 = np.arange(12).reshape((2, 6));
            var y2 = (int[,])x2.ToMultiDimArray<int>();
            Assert.AreEqual(x2[0, 5], y2[0, 5]);

            var x3 = np.arange(12).reshape((2, 2, 3));
            var y3 = (int[,,])x3.ToMultiDimArray<int>();
            Assert.AreEqual(x3[0, 1, 2], y3[0, 1, 2]);
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

        [TestMethod]
        public void to_numpy_string()
        {
            var nd = np.arange(10 * 10 * 10 * 10).reshape((10, 10, 10, 10));
            var str = NDArrayRender.ToString(nd);
            Assert.AreEqual("array([[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],", str.Substring(0, 40));
            Assert.AreEqual("[9990, 9991, 9992, 9993, 9994, 9995, 9996, 9997, 9998, 9999]]]])", str.Substring(str.Length - 64));
        }
    }
}
