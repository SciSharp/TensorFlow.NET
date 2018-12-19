using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class VersionTest
    {
        [TestMethod]
        public void GetVersion()
        {
            var ver = tf.VERSION;
            Assert.IsTrue(ver.StartsWith("1."));
        }
    }
}
