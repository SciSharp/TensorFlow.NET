using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class PythonBaseTests : PythonTest
    {
        [TestMethod]
        public void hasattr_getattr()
        {
            var s1 = "Tensorflow v0.1";
            var f = "Tensorflow";
            var r = "Tensorflow.NET";
            var res = s1.Replace(f, r);

            // Test 1
            Assert.IsTrue(hasattr(s1, "Replace"));

            // Test 2
            var o = getattr( s1, "Replace", typeof(string), typeof(string));
            Assert.AreEqual(res, o(f, r));

            // Test 3
            var l = getattr(s1, "Length");
            Assert.AreEqual(s1.Length, l());

        }
    }
}

