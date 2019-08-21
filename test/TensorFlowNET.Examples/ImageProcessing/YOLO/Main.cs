using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    /// <summary>
    /// Implementation of YOLO v3 object detector in Tensorflow
    /// https://github.com/YunYang1994/tensorflow-yolov3
    /// </summary>
    public class Main : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = false;

        public string Name => "YOLOv3";

        Dictionary<int, string> classes;
        Config config;

        Tensor input_data;
        Tensor label_sbbox;
        Tensor label_mbbox;
        Tensor label_lbbox;
        Tensor true_sbboxes;
        Tensor true_mbboxes;
        Tensor true_lbboxes;
        Tensor trainable;

        public bool Run()
        {
            PrepareData();

            var graph = IsImportingGraph ? ImportGraph() : BuildGraph();

            using (var sess = tf.Session(graph))
            {
                Train(sess);
            }

            return true;
        }

        public void Train(Session sess)
        {

        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            var graph = new Graph().as_default();

            tf_with(tf.name_scope("define_input"), scope =>
            {
                input_data = tf.placeholder(dtype: tf.float32, name: "input_data");
                label_sbbox = tf.placeholder(dtype: tf.float32, name: "label_sbbox");
                label_mbbox = tf.placeholder(dtype: tf.float32, name: "label_mbbox");
                label_lbbox = tf.placeholder(dtype: tf.float32, name: "label_lbbox");
                true_sbboxes = tf.placeholder(dtype: tf.float32, name: "sbboxes");
                true_mbboxes = tf.placeholder(dtype: tf.float32, name: "mbboxes");
                true_lbboxes = tf.placeholder(dtype: tf.float32, name: "lbboxes");
                trainable = tf.placeholder(dtype: tf.@bool, name: "training");
            });

            tf_with(tf.name_scope("define_loss"), scope =>
            {
                //model = new YOLOv3(input_data, trainable);
            });

            return graph;
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public void Predict(Session sess)
        {
            throw new NotImplementedException();
        }

        public void PrepareData()
        {
            config = new Config(Name);

            string dataDir = Path.Combine(Name, "data");
            Directory.CreateDirectory(dataDir);

            classes = new Dictionary<int, string>();
            foreach (var line in File.ReadAllLines(config.CLASSES))
                classes[classes.Count] = line;
        }
    }
}
