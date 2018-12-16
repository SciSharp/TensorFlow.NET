using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlowNET.Core;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class OperationsTest
    {
        [TestMethod]
        public void constant()
        {
            tf.constant(4.0);
        }

        [TestMethod]
        public void placeholder()
        {
            var x = tf.placeholder(tf.float32, shape: new TensorShape(1024, 1024));
        }
    }
}
