using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class TrainSaverTest
    {
        public void ExportGraph()
        {
            var v = tf.Variable(0, name: "my_variable");
            var sess = tf.Session();
            tf.train.write_graph(sess.graph, "/tmp/my-model", "train1.pbtxt");
        }

        public void ImportGraph()
        {
            var sess = tf.Session();
            var new_saver = tf.train.import_meta_graph("C:/tmp/my-model.meta");

            //tf.train.export_meta_graph(filename: "linear_regression.meta.bin");
            // import meta
            /*tf.train.import_meta_graph("linear_regression.meta.bin");

            var cost = graph.OperationByName("truediv").output;
            var pred = graph.OperationByName("Add").output;
            var optimizer = graph.OperationByName("GradientDescent");
            var X = graph.OperationByName("Placeholder").output;
            var Y = graph.OperationByName("Placeholder_1").output;
            var W = graph.OperationByName("weight").output;
            var b = graph.OperationByName("bias").output;*/

            /*var text = JsonConvert.SerializeObject(graph, new JsonSerializerSettings
            {
                Formatting = Formatting.Indented
            });*/
        }

        public void ImportSavedModel()
        {
            Session.LoadFromSavedModel("mobilenet");
        }

        public void ImportGraphDefFromPbFile()
        {
            var g = new Graph();
            var status = g.Import("mobilenet/saved_model.pb");
        }

        public void Save1()
        {
            var w1 = tf.Variable(0, name: "save1");

            var init_op = tf.global_variables_initializer();

            // Add ops to save and restore all the variables.
            var saver = tf.train.Saver();

            var sess = tf.Session();
            sess.run(init_op);

            // Save the variables to disk.
            var save_path = saver.save(sess, "/tmp/model1.ckpt");
            Console.WriteLine($"Model saved in path: {save_path}");
        }

        public void Save2()
        {
            var v1 = tf.compat.v1.get_variable("v1", shape: new Shape(3), initializer: tf.zeros_initializer);
            var v2 = tf.compat.v1.get_variable("v2", shape: new Shape(5), initializer: tf.zeros_initializer);

            var inc_v1 = v1.assign(v1.AsTensor() + 1.0f);
            var dec_v2 = v2.assign(v2.AsTensor() - 1.0f);

            // Add an op to initialize the variables.
            var init_op = tf.global_variables_initializer();

            // Add ops to save and restore all the variables.
            var saver = tf.train.Saver();

            var sess = tf.Session();
            sess.run(init_op);
            // o some work with the model.
            inc_v1.op.run();
            dec_v2.op.run();

            // Save the variables to disk.
            var save_path = saver.save(sess, "/tmp/model2.ckpt");
            Console.WriteLine($"Model saved in path: {save_path}");
        }
    }
}
