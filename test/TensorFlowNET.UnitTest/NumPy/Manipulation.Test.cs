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
    /// https://numpy.org/doc/stable/reference/routines.array-manipulation.html
    /// </summary>
    [TestClass]
    public class ManipulationTest : EagerModeTestBase
    {
        [TestMethod]
        public void expand_dims()
        {
            var x = np.array(new[] { 1, 2 });
            var y = np.expand_dims(x, axis: 0);
            Assert.AreEqual(y.shape, (1, 2));

            y = np.expand_dims(x, axis: 1);
            Assert.AreEqual(y.shape, (2, 1));
        }

        [TestMethod]
        public void moveaxis()
        {
            var x = np.zeros((3, 4, 5));
            var y = np.moveaxis(x, 0, -1);
            Assert.AreEqual(y.shape, (4, 5, 3));

            y = np.moveaxis(x, (0, 1), (-1, -2));
            Assert.AreEqual(y.shape, (5, 4, 3));

            y = np.moveaxis(x, (0, 1, 2), (-1, -2, -3));
            Assert.AreEqual(y.shape, (5, 4, 3));
        }
    }
}
