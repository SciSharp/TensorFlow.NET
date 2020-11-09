using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;
using static Tensorflow.KerasExt;

namespace TensorFlowNET.UnitTest
{
    public class EagerModeTestBase : PythonTest
    {
        protected LayersApi layers = keras.layers;

        [TestInitialize]
        public void TestInit()
        {
            if (!tf.executing_eagerly())
                tf.enable_eager_execution();
        }

        [TestCleanup]
        public void TestClean()
        {
        }
    }
}
