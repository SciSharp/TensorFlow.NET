using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
using System;
using Tensorflow;

namespace TensorFlowNET.UnitTest.control_flow_ops_test
{
    /// <summary>
    /// excerpt of tensorflow/python/framework/ops/control_flow_ops_test.py
    /// </summary>
    [TestClass]
    public class CondTestCases : PythonTest
    {
        [TestMethod]
        public void testCondTrue()
        {
            var graph = tf.Graph().as_default();
            // tf.train.import_meta_graph("cond_test.meta");
            var json = JsonConvert.SerializeObject(graph._nodes_by_name, Formatting.Indented);

            with(tf.Session(graph), sess =>
            {
                var x = tf.constant(2, name: "x"); // graph.get_operation_by_name("Const").output; 
                var y = tf.constant(5, name: "y"); // graph.get_operation_by_name("Const_1").output;
                var pred = tf.less(x, y); // graph.get_operation_by_name("Less").output;

                Func<ITensorOrOperation> if_true = delegate
                {
                    return tf.constant(2, name: "t2");
                };

                Func<ITensorOrOperation> if_false = delegate
                {
                    return tf.constant(5, name: "f5");
                };

                var z = control_flow_ops.cond(pred, if_true, if_false); // graph.get_operation_by_name("cond/Merge").output

                json = JsonConvert.SerializeObject(graph._nodes_by_name, Formatting.Indented);
                int result = z.eval(sess);
                assertEquals(result, 2);
            });
        }

        [TestMethod]
        public void testCondFalse()
        {
            /* python
             * import tensorflow as tf
               from tensorflow.python.framework import ops

                def if_true():
                    return tf.math.multiply(x, 17)
                def if_false():
                    return tf.math.add(y, 23)

                with tf.Session() as sess:
                    x = tf.constant(2)
                    y = tf.constant(1)
                    pred = tf.math.less(x,y)
                    z = tf.cond(pred, if_true, if_false)
                    result = z.eval()

                print(result == 24) */

            var graph = tf.Graph().as_default();
            //tf.train.import_meta_graph("cond_test.meta");
            //var json = JsonConvert.SerializeObject(graph._nodes_by_name, Formatting.Indented);

            with(tf.Session(), sess =>
            {
                var x = tf.constant(2, name: "x");
                var y = tf.constant(1, name: "y");
                var pred = tf.less(x, y);

                Func<ITensorOrOperation> if_true = delegate
                {
                    return tf.constant(2, name: "t2");
                };

                Func<ITensorOrOperation> if_false = delegate
                {
                    return tf.constant(1, name: "f1");
                };

                var z = control_flow_ops.cond(pred, if_true, if_false);

                var json1 = JsonConvert.SerializeObject(graph._nodes_by_name, Formatting.Indented);
                int result = z.eval(sess);
                assertEquals(result, 1);
            });
        }

        // NOTE: all other test python test cases of this class are either not needed due to strong typing or dest a deprecated api

    }
}
