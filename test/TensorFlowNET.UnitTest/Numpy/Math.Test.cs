using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.NumPy;

namespace TensorFlowNET.UnitTest.Numpy
{
    /// <summary>
    /// https://numpy.org/doc/stable/reference/generated/numpy.prod.html
    /// </summary>
    [TestClass]
    public class NumpyMathTest : EagerModeTestBase
    {
        [TestMethod]
        public void prod()
        {
            var p = np.prod(1.0, 2.0);
            Assert.AreEqual(p, 2.0);

            p = np.prod(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            Assert.AreEqual(p, 24.0);
        }
    }
}
