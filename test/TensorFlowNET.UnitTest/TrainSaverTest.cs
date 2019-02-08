using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class TrainSaverTest : Python
    {
        [TestMethod]
        public void Save()
        {
            var v1 = tf.get_variable("v1", shape: new TensorShape(3), initializer: tf.zeros_initializer);
            var v2 = tf.get_variable("v2", shape: new TensorShape(5), initializer: tf.zeros_initializer);

            var inc_v1 = v1.assign(v1 + 1.0f);
            var dec_v2 = v2.assign(v2 - 1.0f);

            // Add an op to initialize the variables.
            var init_op = tf.global_variables_initializer();

            // Add ops to save and restore all the variables.
            var saver = tf.train.Saver();

            with<Session>(tf.Session(), sess =>
            {
                sess.run(init_op);
            });
        }
    }
}
