using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class LinalgTest
    {
        [TestMethod]
        public void EyeTest()
        {
            var tensor = tf.linalg.eye(3);

            Assert.AreEqual((3, 3), tensor.TensorShape);

            Assert.AreEqual(0.0f, (float)tensor[2, 0]);
            Assert.AreEqual(0.0f, (float)tensor[2, 1]);
            Assert.AreEqual(1.0f, (float)tensor[2, 2]);
        }
    }
}
