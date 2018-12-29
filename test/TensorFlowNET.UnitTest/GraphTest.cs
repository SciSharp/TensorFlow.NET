using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class GraphTest
    {
        [TestMethod]
        public void Graph()
        {
            var s = new Status();
            var graph = tf.get_default_graph();

            // Make a placeholder operation.
            var feed = c_test_util.Placeholder(graph, s);
            Assert.AreEqual("feed", feed.name);
            Assert.AreEqual("Placeholder", feed.optype);
            //Assert.AreEqual("", feed.device);
            Assert.AreEqual(1, feed.NumOutputs);
            Assert.AreEqual(TF_DataType.TF_INT32, feed.OutputType);
            Assert.AreEqual(1, feed.OutputListLength);
            Assert.AreEqual(0, feed.NumInputs);
            Assert.AreEqual(0, feed.NumConsumers);
            Assert.AreEqual(0, feed.NumControlInputs);
            Assert.AreEqual(0, feed.NumControlOutputs);

            AttrValue attr_value = null;
            c_test_util.GetAttrValue(feed, "dtype", ref attr_value, s);
        }
    }
}
