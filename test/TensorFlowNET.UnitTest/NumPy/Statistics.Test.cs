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
    /// https://numpy.org/doc/stable/reference/routines.statistics.html
    /// </summary>
    [TestClass]
    public class StatisticsTest : EagerModeTestBase
    {
        [TestMethod]
        public void average()
        {
            var data = np.arange(1, 5);
            var avg = np.average(data);
            Assert.AreEqual(avg, 2.5);

            data = np.arange(6).reshape((3, 2));
            avg = np.average(data, axis: 1);
            assertAllEqual(avg.ToArray<double>(), new[] { 0.5, 2.5, 4.5 });

            // avg = np.average(data, axis: 1, weights: new[] { 1.0 / 4, 3.0 / 4 });
            // assertAllEqual(avg.ToArray<double>(), new[] { 0.75, 2.75, 4.75 });
        }
    }
}
