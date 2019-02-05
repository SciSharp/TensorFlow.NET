using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class NameScopeTest : Python
    {
        Graph g = ops.get_default_graph();
        string name = "";

        [TestMethod]
        public void NestedNameScope()
        {
            with<ops.name_scope>(new ops.name_scope("scope1"), scope1 =>
            {
                name = scope1;
                Assert.AreEqual("scope1", g._name_stack);
                Assert.AreEqual("scope1/", name);

                var const1 = tf.constant(1.0);
                Assert.AreEqual("scope1/Const:0", const1.name);

                with<ops.name_scope>(new ops.name_scope("scope2"), scope2 =>
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

            Assert.AreEqual("", g._name_stack);
        }
    }
}
