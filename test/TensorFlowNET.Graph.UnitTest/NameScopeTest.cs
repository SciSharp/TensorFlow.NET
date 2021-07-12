using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class NameScopeTest : GraphModeTestBase
    {
        string name = "";

        [TestMethod]
        public void NestedNameScope()
        {
            Graph g = tf.Graph().as_default();

            tf_with(new ops.NameScope("scope1"), scope1 =>
            {
                name = scope1;
                Assert.AreEqual("scope1", g._name_stack);
                Assert.AreEqual("scope1/", name);

                var const1 = tf.constant(1.0);
                Assert.AreEqual("scope1/Const:0", const1.name);

                tf_with(new ops.NameScope("scope2"), scope2 =>
                {
                    name = scope2;
                    Assert.AreEqual("scope1/scope2", g._name_stack);
                    Assert.AreEqual("scope1/scope2/", name);

                    var const2 = tf.constant(2.0);
                    Assert.AreEqual("scope1/scope2/Const:0", const2.name);
                });

                Assert.AreEqual("scope1", g._name_stack);
                var const3 = tf.constant(2.0);
                Assert.AreEqual("scope1/Const_1:0", const3.name);
            });

            g.Exit();

            Assert.AreEqual("", g._name_stack);
        }

        [TestMethod, Ignore("Unimplemented Usage")]
        public void NestedNameScope_Using()
        {
            Graph g = tf.Graph().as_default();

            using (var name = new ops.NameScope("scope1"))
            {
                Assert.AreEqual("scope1", g._name_stack);
                Assert.AreEqual("scope1/", name);

                var const1 = tf.constant(1.0);
                Assert.AreEqual("scope1/Const:0", const1.name);

                using (var name2 = new ops.NameScope("scope2"))
                {
                    Assert.AreEqual("scope1/scope2", g._name_stack);
                    Assert.AreEqual("scope1/scope2/", name);

                    var const2 = tf.constant(2.0);
                    Assert.AreEqual("scope1/scope2/Const:0", const2.name);
                }

                Assert.AreEqual("scope1", g._name_stack);
                var const3 = tf.constant(2.0);
                Assert.AreEqual("scope1/Const_1:0", const3.name);
            };

            g.Exit();

            Assert.AreEqual("", g._name_stack);
        }
    }
}
