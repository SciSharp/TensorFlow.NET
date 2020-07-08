using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlowNET.UnitTest;
using static Tensorflow.Binding;

namespace Tensorflow.UnitTest
{
    public class EagerModeTestBase : PythonTest
    {
        [TestInitialize]
        public void TestInit()
        {
            tf.enable_eager_execution();
        }

        [TestCleanup]
        public void TestClean()
        {
        }
    }
}
