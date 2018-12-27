using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class VariableTest
    {
        [TestMethod]
        public void Creating()
        {
            var mammal = tf.Variable("Elephant", tf.chars);
        }
    }
}
