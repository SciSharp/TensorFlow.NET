using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using NumSharp.Core;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class VariableTest : Python
    {
        [TestMethod]
        public void Initializer()
        {
            var x = tf.Variable(10, name: "x");
            
            using (var session = tf.Session())
            {
                session.run(x.initializer);
                var result = session.run(x);
                Assert.AreEqual(10, (int)result);
            }
        }

        [TestMethod]
        public void StringVar()
        {
            var mammal1 = tf.Variable("Elephant", "var1", tf.chars);
            var mammal2 = tf.Variable("Tiger");
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
                int result = session.run(y);
                Assert.AreEqual(result, 4);
            }
        }

        [TestMethod]
        public void Assign()
        {
            var v1 = tf.get_variable("v1", shape: new TensorShape(3), initializer: tf.zeros_initializer);

            var inc_v1 = v1.assign(v1 + 1.0f);

            // Add an op to initialize the variables.
            var init_op = tf.global_variables_initializer();

            with<Session>(tf.Session(), sess =>
            {
                sess.run(init_op);
                // o some work with the model.
                inc_v1.op.run();
            });
        }

        /// <summary>
        /// https://databricks.com/tensorflow/variables
        /// </summary>
        [TestMethod]
        public void Add()
        {
            int result = 0;
            Tensor x = tf.Variable(10, name: "x");

            var model = tf.global_variables_initializer();
            using (var session = tf.Session())
            {
                session.run(model);
                for(int i = 0; i < 5; i++)
                {
                    x = x + 1;
                    result = session.run(x);
                    print(result);
                }
            }

            Assert.AreEqual(15, result);
        }
    }
}
