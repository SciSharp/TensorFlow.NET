using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorFlowNET.UnitTest;
using static Tensorflow.Binding;

namespace Tensorflow.UnitTest
{
    public class GraphModeTestBase : PythonTest
    {
        [TestInitialize]
        public void TestInit()
        {
            tf.compat.v1.disable_eager_execution();
        }

        [TestCleanup]
        public void TestClean()
        {
            tf.enable_eager_execution();
        }
    }
}
