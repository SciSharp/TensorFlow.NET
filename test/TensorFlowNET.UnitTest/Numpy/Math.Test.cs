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
    /// https://numpy.org/doc/stable/reference/generated/numpy.prod.html
    /// </summary>
    [TestClass]
    public class MathTest : EagerModeTestBase
    {
        [TestMethod]
        public void prod()
        {
            var p = np.prod(1.0, 2.0);
            Assert.AreEqual(p, 2.0);

            p = np.prod(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            Assert.AreEqual(p, 24.0);

            p = np.prod(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }, axis: 1);
            Assert.AreEqual(p.shape, 2);
            Assert.IsTrue(Equal(p.ToArray<double>(), new[] { 2.0, 12.0 }));
        }

        [TestMethod]
        public void astype()
        {
            var x = np.array(new byte[] { 1, 100, 200 });
            var x1 = x.astype(np.float32);
            Assert.AreEqual(x1[2], 200f);
        }

        [TestMethod]
        public void divide()
        {
            var x = np.array(new float[] { 1, 100, 200 });
            var y = x / 2;
            Assert.AreEqual(y.dtype, np.float32);
        }

        [TestMethod]
        public void sin()
        {
            var x = np.sin(np.pi / 2);
            Assert.AreEqual(x, 1d);
        }

        [TestMethod]
        public void cos()
        {
            var x = np.cos(np.pi / 2);
            Assert.AreEqual(x, 6.123233995736766e-17);
        }

        [TestMethod]
        public void power()
        {
            var x = np.arange(6);
            var y = np.power(x, 3);
            Assert.AreEqual(y, new[] { 0, 1, 8, 27, 64, 125 });
        }
        [TestMethod]
        public void square()
        {
            var x = np.arange(6);
            var y = np.square(x);
            Assert.AreEqual(y, new[] { 0, 1, 4, 9, 16, 25 });
        }
        [TestMethod]
        public void dotproduct()
        {
            var x1 = new NDArray(new[] { 1, 2, 3 });
            var x2 = new NDArray(new[] { 4, 5, 6 });
            double result1 = np.dot(x1, x2);
            NDArray y1 = new float[,] {
                                    { 1.0f, 2.0f, 3.0f },
                                    { 4.0f, 5.1f,6.0f },
                                    { 4.0f, 5.1f,6.0f }
                                };
            NDArray y2 = new float[,] {
                                    { 3.0f, 2.0f, 1.0f },
                                    { 6.0f, 5.1f, 4.0f },
                                    { 6.0f, 5.1f, 4.0f }
                                };
            double result2 = np.dot(y1, y2);
            Assert.AreEqual(result1, 32);
            Assert.AreEqual(Math.Round(result2, 2), 158.02);
        }
        [TestMethod]
       public void maximum()
        {
            var x1 = new NDArray(new[,] { { 1, 2, 3 }, { 4, 5.1, 6 } });
            var x2 = new NDArray(new[,] { { 3, 2, 1 }, { 6, 5.1, 4 } });
            var y0 = np.maximum(x1,x2);
            var y1 = np.maximum(x1, x2, axis: 0);
            var y2 = np.maximum(x1, x2, axis: 1);
            var y3 = new NDArray(new[,] { { 3, 2, 3 }, { 6, 5.1, 6 } });
            var y4 = new NDArray(new[] {  6, 5.1, 6 });
            var y5 = new NDArray(new[] { 3.0, 6 });
            Assert.AreEqual(y0, y3);
            Assert.AreEqual(y1, y4);
            Assert.AreEqual(y2, y5);
        }
    }
}
