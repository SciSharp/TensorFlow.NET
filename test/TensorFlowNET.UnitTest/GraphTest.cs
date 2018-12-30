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
        /// <summary>
        /// Port from c_api_test.cc
        /// `TEST(CAPI, Graph)`
        /// </summary>
        [TestMethod]
        public void c_api_Graph()
        {
            var s = new Status();
            var graph = new Graph();

            // Make a placeholder operation.
            var feed = c_test_util.Placeholder(graph, s);
            Assert.AreEqual("feed", feed.name);
            Assert.AreEqual("Placeholder", feed.optype);
            Assert.AreEqual("", feed.device);
            Assert.AreEqual(1, feed.NumOutputs);
            Assert.AreEqual(TF_DataType.TF_INT32, feed.OutputType);
            Assert.AreEqual(1, feed.OutputListLength);
            Assert.AreEqual(0, feed.NumInputs);
            Assert.AreEqual(0, feed.NumConsumers);
            Assert.AreEqual(0, feed.NumControlInputs);
            Assert.AreEqual(0, feed.NumControlOutputs);

            AttrValue attr_value = null;
            Assert.IsTrue(c_test_util.GetAttrValue(feed, "dtype", ref attr_value, s));
            Assert.AreEqual(attr_value.Type, DataType.DtInt32);

            // Test not found errors in TF_Operation*() query functions.
            Assert.AreEqual(-1, c_api.TF_OperationOutputListLength(feed, "bogus", s));
            Assert.AreEqual(TF_Code.TF_INVALID_ARGUMENT, s.Code);
            Assert.IsFalse(c_test_util.GetAttrValue(feed, "missing", ref attr_value, s));
            Assert.AreEqual("Operation 'feed' has no attr named 'missing'.", s.Message);

            // Make a constant oper with the scalar "3".
            var three = c_test_util.ScalarConst(3, graph, s);

            // Add oper.
            var add = c_test_util.Add(feed, three, graph, s);
        }
    }
}
