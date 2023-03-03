using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class VariableTest : GraphModeTestBase
    {
        [TestMethod]
        public void InitVariable()
        {
            var v = tf.Variable(new[] { 1, 2 });
            var init = tf.compat.v1.global_variables_initializer();

            var sess = tf.compat.v1.Session();
            sess.run(init);
            // Usage passing the session explicitly.
            print(v.eval(sess));
            // Usage with the default session.  The 'with' block
            // above makes 'sess' the default session.
            print(v.eval());
        }
    }
}
