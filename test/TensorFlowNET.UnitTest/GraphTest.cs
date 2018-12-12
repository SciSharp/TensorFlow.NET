using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using tf = TensorFlowNET.Core.Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class GraphTest
    {
        [TestMethod]
        public void ConstructGraph()
        {
            var g = tf.Graph();
        }
    }
}
