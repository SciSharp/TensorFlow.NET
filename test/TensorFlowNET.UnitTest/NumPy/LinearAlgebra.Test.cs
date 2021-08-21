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
    public class LinearAlgebraTest : EagerModeTestBase
    {
        [TestMethod]
        public void lstsq()
        {

        }

        [TestMethod]
        public void norm()
        {
            var x = np.arange(9) - 4;
            var y = x.reshape((3, 3));
            var norm = np.linalg.norm(y);
        }
    }
}
