using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using static Tensorflow.TextApi;

namespace TensorFlowNET.UnitTest.Text
{
    [TestClass]
    public class TokenizerTest
    {
        [TestMethod]
        public void Tokenize()
        {
            var docs = tf.constant(new[] { "Everything not saved will be lost." });
        }
    }
}
