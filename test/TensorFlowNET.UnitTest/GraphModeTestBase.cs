using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorFlowNET.UnitTest;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

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
            keras.backend.clear_session();
            tf.enable_eager_execution();
        }
    }
}
