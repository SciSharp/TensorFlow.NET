using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    /// <summary>
    /// Use as base class for test classes to get additional assertions
    /// </summary>
    public class PythonTest : Python
    {
        public void assertItemsEqual(ICollection expected, ICollection given)
        {
            Assert.IsNotNull(expected);
            Assert.IsNotNull(given);
            var e = expected.OfType<object>().ToArray();
            var g = given.OfType<object>().ToArray();
            Assert.AreEqual(e.Length, g.Length, $"The collections differ in length expected {e.Length} but got {g.Length}");
            for(int i=0; i<e.Length; i++)
                Assert.AreEqual(e[i], g[i], $"Items differ at index {i}, expected {e[i]} but got {g[i]}");
        }

        public void assertEqual(object given, object expected)
        {
            Assert.AreEqual(expected, given);
        }

        protected PythonTest self { get => this; }
    }
}
