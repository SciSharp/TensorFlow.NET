using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class VariableTest
    {
        [TestMethod]
        public void Initializer()
        {
            var x = tf.Variable(10, name: "x");
            
            using (var session = tf.Session())
            {
                session.run(x.initializer);
                var result = session.run(x);
                Assert.AreEqual(10, (int)result[0]);
            }
        }

        [TestMethod]
        public void StringVar()
        {
            var mammal1 = tf.Variable("Elephant", name: "var1", dtype: tf.chars);
            var mammal2 = tf.Variable("Tiger");
        }

        /// <summary>
        /// https://www.tf.org/api_docs/python/tf/variable_scope
        /// how to create a new variable
        /// </summary>
        [TestMethod]
        public void VarCreation()
        {
            tf.Graph().as_default();
            tf_with(tf.variable_scope("foo"), delegate
            {
                tf_with(tf.variable_scope("bar"), delegate
                {
                    var v = tf.get_variable("v", new TensorShape(1));
                    Assert.AreEqual(v.name, "foo/bar/v:0");
                });
            });
        }

        /// <summary>
        /// how to reenter a premade variable scope safely
        /// </summary>
        [TestMethod]
        public void ReenterVariableScope()
        {
            tf.Graph().as_default();
            variable_scope vs = null;
            tf_with(tf.variable_scope("foo"), v => vs = v);

            // Re-enter the variable scope.
            tf_with(tf.variable_scope(vs, auxiliary_name_scope: false), v =>
            {
                var vs1 = (VariableScope)v;
                // Restore the original name_scope.
                tf_with(tf.name_scope(vs1.original_name_scope), delegate
                {
                    var v1 = tf.get_variable("v", new TensorShape(1));
                    Assert.AreEqual(v1.name, "foo/v:0");
                    var c1 = tf.constant(new int[] { 1 }, name: "c");
                    Assert.AreEqual(c1.name, "foo/c:0");
                });
            });
        }

        [TestMethod]
        public void ScalarVar()
        {
            var x = tf.constant(3, name: "x");
            var y = tf.Variable(x + 1, name: "y");

            var model = tf.global_variables_initializer();

            using (var session = tf.Session())
            {
                session.run(model);
                int result = session.run(y)[0];
                Assert.AreEqual(result, 4);
            }
        }

        [TestMethod]
        public void Assign1()
        {
            var graph = tf.Graph().as_default();

            var variable = tf.Variable(31, name: "tree");
            var init = tf.global_variables_initializer();

            var sess = tf.Session(graph);
            sess.run(init);

            var result = sess.run(variable);
            Assert.IsTrue((int)result[0] == 31);

            var assign = variable.assign(12);
            result = sess.run(assign);
            Assert.IsTrue((int)result[0] == 12);
        }

        [TestMethod]
        public void Assign2()
        {
            var v1 = tf.Variable(10.0f, name: "v1"); //tf.get_variable("v1", shape: new TensorShape(3), initializer: tf.zeros_initializer);
            var inc_v1 = v1.assign(v1 + 1.0f);

            // Add an op to initialize the variables.
            var init_op = tf.global_variables_initializer();

            using (var sess = tf.Session())
            {
                sess.run(init_op);
                // o some work with the model.
                inc_v1.op.run();
            }
        }

        /// <summary>
        /// https://databricks.com/tensorflow/variables
        /// </summary>
        [TestMethod]
        public void Add()
        {
            tf.Graph().as_default();
            int result = 0;
            Tensor x = tf.Variable(10, name: "x");

            var init_op = tf.global_variables_initializer();
            using (var session = tf.Session())
            {
                session.run(init_op);
                for(int i = 0; i < 5; i++)
                {
                    x = x + 1;
                    result = session.run(x)[0];
                    print(result);
                }
            }

            Assert.AreEqual(15, result);
        }
    }
}
