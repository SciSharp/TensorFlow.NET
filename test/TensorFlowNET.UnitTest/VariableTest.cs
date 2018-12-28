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
        public void StringVar()
        {
            var mammal1 = tf.Variable("Elephant", tf.chars);
            var mammal2 = tf.Variable("Tiger");
        }

        [TestMethod]
        public void ScalarVar()
        {
            var x = tf.Variable(3);
            var y = tf.Variable(6f);
        }
    }
}
