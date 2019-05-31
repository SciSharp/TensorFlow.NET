using NumSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Console = Colorful.Console;
using System.Linq;
using System.Net;
using System.Text;
using Tensorflow;
using System.Drawing;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Inception v3 is a widely-used image recognition model 
    /// that has been shown to attain greater than 78.1% accuracy on the ImageNet dataset. 
    /// The model is the culmination of many ideas developed by multiple researchers over the years.
    /// </summary>
    public class ImageRecognitionInception : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Image Recognition Inception";
        public bool IsImportingGraph { get; set; } = false;


        string dir = "ImageRecognitionInception";
        string pbFile = "tensorflow_inception_graph.pb";
        string labelFile = "imagenet_comp_graph_label_strings.txt";
        List<NDArray> file_ndarrays = new List<NDArray>();

        public bool Run()
        {
            PrepareData();
           
            var graph = new Graph().as_default();
            //import GraphDef from pb file
            graph.Import(Path.Join(dir, pbFile));

            var input_name = "input";
            var output_name = "output";

            var input_operation = graph.OperationByName(input_name);
            var output_operation = graph.OperationByName(output_name);

            var labels = File.ReadAllLines(Path.Join(dir, labelFile));
            var result_labels = new List<string>();
            var sw = new Stopwatch();

            with(tf.Session(graph), sess =>
            {
                foreach (var nd in file_ndarrays)
                {
                    sw.Restart();

                    var results = sess.run(output_operation.outputs[0], new FeedItem(input_operation.outputs[0], nd));
                    results = np.squeeze(results);
                    int idx = np.argmax(results);

                    Console.WriteLine($"{labels[idx]} {results[idx]} in {sw.ElapsedMilliseconds}ms", Color.Tan);
                    result_labels.Add(labels[idx]);
                }
            });
            
            return result_labels.Contains("military uniform");
        }

        private NDArray ReadTensorFromImageFile(string file_name,
                                int input_height = 224,
                                int input_width = 224,
                                int input_mean = 117,
                                int input_std = 1)
        {
            return with(tf.Graph().as_default(), graph =>
            {
                var file_reader = tf.read_file(file_name, "file_reader");
                var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
                var cast = tf.cast(decodeJpeg, tf.float32);
                var dims_expander = tf.expand_dims(cast, 0);
                var resize = tf.constant(new int[] { input_height, input_width });
                var bilinear = tf.image.resize_bilinear(dims_expander, resize);
                var sub = tf.subtract(bilinear, new float[] { input_mean });
                var normalized = tf.divide(sub, new float[] { input_std });

                return with(tf.Session(graph), sess => sess.run(normalized));
            });
        }

        public void PrepareData()
        {
            Directory.CreateDirectory(dir);

            // get model file
            string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";

            Utility.Web.Download(url, dir, "inception5h.zip");

            Utility.Compress.UnZip(Path.Join(dir, "inception5h.zip"), dir);

            // download sample picture
            Directory.CreateDirectory(Path.Join(dir, "img"));
            url = $"https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/grace_hopper.jpg";
            Utility.Web.Download(url, Path.Join(dir, "img"), "grace_hopper.jpg");

            url = $"https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/data/shasta-daisy.jpg";
            Utility.Web.Download(url, Path.Join(dir, "img"), "shasta-daisy.jpg");

            // load image file
            var files = Directory.GetFiles(Path.Join(dir, "img"));
            for (int i = 0; i < files.Length; i++)
            {
                var nd = ReadTensorFromImageFile(files[i]);
                file_ndarrays.Add(nd);
            }
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public bool Train()
        {
            throw new NotImplementedException();
        }

        public bool Predict()
        {
            throw new NotImplementedException();
        }
    }
}
