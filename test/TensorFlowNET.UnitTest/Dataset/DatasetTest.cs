using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.UnitTest;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Dataset
{
    [TestClass]
    public class DatasetTest : EagerModeTestBase
    {
        [TestMethod]
        public void Range()
        {
            int iStep = 0;
            long value = 0;

            var dataset = tf.data.Dataset.range(3);
            foreach(var (step, item) in enumerate(dataset))
            {
                Assert.AreEqual(iStep, step);
                iStep++;

                Assert.AreEqual(value, (long)item.Item1);
                value++;
            }
        }

        [TestMethod]
        public void Prefetch()
        {
            int iStep = 0;
            long value = 1;

            var dataset = tf.data.Dataset.range(1, 5, 2);
            dataset = dataset.prefetch(2);

            foreach (var (step, item) in enumerate(dataset))
            {
                Assert.AreEqual(iStep, step);
                iStep++;

                Assert.AreEqual(value, (long)item.Item1);
                value += 2;
            }
        }
    }
}
