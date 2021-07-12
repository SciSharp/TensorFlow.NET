using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    public class GraphModeTestBase : PythonTest
    {
        protected Graph graph;
        [TestInitialize]
        public void TestInit()
        {
            tf.compat.v1.disable_eager_execution();
            graph = tf.Graph().as_default();
        }

        [TestCleanup]
        public void TestClean()
        {
            graph.Exit();
        }
    }
}
