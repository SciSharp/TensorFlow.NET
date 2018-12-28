using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class PlaceholderTest
    {
        [TestMethod]
        public void placeholder()
        {
            var x = tf.placeholder(tf.float32);
        }
    }
}
