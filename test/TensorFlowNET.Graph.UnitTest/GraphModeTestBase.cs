using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    public class GraphModeTestBase : PythonTest
    {
        [TestInitialize]
        public void TestInit()
        {
            tf.compat.v1.disable_eager_execution();
        }
    }
}
