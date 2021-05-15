using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class OutputTest
    {
        [TestMethod]
        public void OutputRedirectTest()
        {
            using var newOutput = new System.IO.StringWriter();
            tf_output_redirect = newOutput;
            var model = keras.Sequential();
            model.add(keras.Input(shape: 16));
            model.summary();
            string output = newOutput.ToString();
            Assert.IsTrue(output.StartsWith("Model: sequential"));
            tf_output_redirect = null; // don't forget to change it to null !!!!
        }

        [TestMethod]
        public void SwitchOutputsTest()
        {
            using var newOutput = new System.IO.StringWriter();
            var model = keras.Sequential();
            model.add(keras.Input(shape: 16));
            model.summary(); // Console.Out

            tf_output_redirect = newOutput; // change to the custom one
            model.summary();
            string firstOutput = newOutput.ToString();
            Assert.IsTrue(firstOutput.StartsWith("Model: sequential"));

            // if tf_output_reditect is StringWriter, calling "set" will make the writer clear.
            tf_output_redirect = null; // null means Console.Out
            model.summary();

            tf_output_redirect = newOutput; // again, to test whether the newOutput is clear.
            model.summary();
            string secondOutput = newOutput.ToString();
            Assert.IsTrue(secondOutput.StartsWith("Model: sequential"));

            Assert.IsTrue(firstOutput == secondOutput);
            tf_output_redirect = null; // don't forget to change it to null !!!!
        }
    }
}
