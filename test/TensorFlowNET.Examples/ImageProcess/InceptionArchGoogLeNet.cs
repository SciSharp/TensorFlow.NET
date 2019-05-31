using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Inception Architecture for Computer Vision
    /// Port from tensorflow\examples\label_image\label_image.py
    /// </summary>
    public class InceptionArchGoogLeNet : IExample
    {
        public bool Enabled { get; set; } = false;
        public string Name => "Inception Arch GoogLeNet";
        public bool IsImportingGraph { get; set; } = false;


        string dir = "label_image_data";
        string pbFile = "inception_v3_2016_08_28_frozen.pb";
        string labelFile = "imagenet_slim_labels.txt";
        string picFile = "grace_hopper.jpg";
        int input_height = 299;
        int input_width = 299;
        int input_mean = 0;
        int input_std = 255;
        string input_name = "import/input";
        string output_name = "import/InceptionV3/Predictions/Reshape_1";

        public bool Run()
        {
            PrepareData();

            var labels = File.ReadAllLines(Path.Join(dir, labelFile));
            
            var nd = ReadTensorFromImageFile(Path.Join(dir, picFile),
                input_height: input_height,
                input_width: input_width,
                input_mean: input_mean,
                input_std: input_std);

            var graph = Graph.ImportFromPB(Path.Join(dir, pbFile));
            var input_operation = graph.get_operation_by_name(input_name);
            var output_operation = graph.get_operation_by_name(output_name);

            var results = with(tf.Session(graph),
                sess => sess.run(output_operation.outputs[0],
                    new FeedItem(input_operation.outputs[0], nd)));

            results = np.squeeze(results);

            var argsort = results.argsort<float>();
            var top_k = argsort.Data<float>()
                .Skip(results.size - 5)
                .Reverse()
                .ToArray();

            foreach (float idx in top_k)
                Console.WriteLine($"{picFile}: {idx} {labels[(int)idx]}, {results[(int)idx]}");

            return true;
        }

        private NDArray ReadTensorFromImageFile(string file_name,
                                int input_height = 299,
                                int input_width = 299,
                                int input_mean = 0,
                                int input_std = 255)
        {
            return with(tf.Graph().as_default(), graph =>
            {
                var file_reader = tf.read_file(file_name, "file_reader");
                var image_reader = tf.image.decode_jpeg(file_reader, channels: 3, name: "jpeg_reader");
                var caster = tf.cast(image_reader, tf.float32);
                var dims_expander = tf.expand_dims(caster, 0);
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
            string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz";
            
            Utility.Web.Download(url, dir, $"{pbFile}.tar.gz");

            Utility.Compress.ExtractTGZ(Path.Join(dir, $"{pbFile}.tar.gz"), dir);

            // download sample picture
            string pic = "grace_hopper.jpg";
            url = $"https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/{pic}";
            Utility.Web.Download(url, dir, pic);
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
