using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    public class ImageRecognition : Python, IExample
    {
        string dir = "ImageRecognition";
        string pbFile = "tensorflow_inception_graph.pb";
        string labelFile = "imagenet_comp_graph_label_strings.txt";
        string picFile = "grace_hopper.jpg";

        public void Run()
        {
            PrepareData();

            var labels = File.ReadAllLines(Path.Join(dir, labelFile));
            var files = Directory.GetFiles(Path.Join(dir, "img"));
            foreach (var file in files)
            {
                var tensor = ReadTensorFromImageFile(file);

                var graph = new Graph().as_default();
                //import GraphDef from pb file
                graph.Import(Path.Join(dir, pbFile));

                var input_name = "input";
                var output_name = "output";

                var input_operation = graph.OperationByName(input_name);
                var output_operation = graph.OperationByName(output_name);

                var idx = 0;
                float propability = 0;
                with(tf.Session(graph), sess =>
                {
                    var results = sess.run(output_operation.outputs[0], new FeedItem(input_operation.outputs[0], tensor));
                    var probabilities = results.Data<float>();
                    for (int i = 0; i < probabilities.Length; i++)
                    {
                        if (probabilities[i] > propability)
                        {
                            idx = i;
                            propability = probabilities[i];
                        }
                    }
                });

                Console.WriteLine($"{picFile}: {labels[idx]} {propability}");
            }
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
        }
    }
}
