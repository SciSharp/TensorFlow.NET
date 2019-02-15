using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Port from tensorflow\examples\label_image\label_image.py
    /// </summary>
    public class LabelImage : Python, IExample
    {
        string dir = "label_image_data";
        string pbFile = "inception_v3_2016_08_28_frozen.pb";
        string labelFile = "imagenet_slim_labels.txt";
        string picFile = "grace_hopper.jpg";
        int input_height = 299;
        int input_width = 299;
        int input_mean = 0;
        int input_std = 255;
        string input_layer = "input";
        string output_layer = "InceptionV3/Predictions/Reshape_1";

        public void Run()
        {
            PrepareData();
            var graph = LoadGraph(Path.Join(dir, pbFile));
            var t = ReadTensorFromImageFile(Path.Join(dir, picFile),
                input_height: input_height,
                input_width: input_width,
                input_mean: input_mean,
                input_std: input_std);

            var input_name = "import/" + input_layer;
            var output_name = "import/" + output_layer;

            var input_operation = graph.get_operation_by_name(input_name);
            var output_operation = graph.get_operation_by_name(output_name);

            NDArray results = null;
            with<Session>(tf.Session(graph), sess =>
            {
                results = sess.run(output_operation.outputs[0], new FeedItem(input_operation.outputs[0], t));
            });

            // equivalent np.squeeze
            results.reshape(results.shape.Where(x => x > 1).ToArray());
            // top_k = results.argsort()[-5:][::-1]
            var top_k = results.Data<int>().Take(5).ToArray();
            var labels = LoadLabels(Path.Join(dir, labelFile));
            foreach (var i in top_k)
                Console.WriteLine($"{labels[i]}, {results[i]}");
        }

        private string[] LoadLabels(string file)
        {
            return File.ReadAllLines(file);
        }

        private Graph LoadGraph(string modelFile)
        {
            var graph = tf.Graph();
            var graph_def = GraphDef.Parser.ParseFrom(File.ReadAllBytes(modelFile));
            importer.import_graph_def(graph_def);
            return graph;
        }

        private NDArray ReadTensorFromImageFile(string file_name,
                                int input_height = 299,
                                int input_width = 299,
                                int input_mean = 0,
                                int input_std = 255)
        {
            string input_name = "file_reader";
            string output_name = "normalized";
            Tensor image_reader = null;

            var file_reader = tf.read_file(file_name, input_name);
            image_reader = tf.image.decode_jpeg(file_reader, channels: 3, name: "jpeg_reader");

            var float_caster = tf.cast(image_reader, tf.float32);
            var dims_expander = tf.expand_dims(float_caster, 0);
            var resized = tf.image.resize_bilinear(dims_expander, new int[] { input_height, input_width });
            var normalized = tf.divide(tf.subtract(resized, new float[] { input_mean }), new float[] { input_std });

            return with<Session, NDArray>(tf.Session(), sess =>
            {
                var result = sess.run(normalized);
                return result;
            });
        }

        private void PrepareData()
        {
            
            Directory.CreateDirectory(dir);

            // get model file
            string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz";
            
            string zipFile = Path.Join(dir, $"{pbFile}.tar.gz");
            Utility.Web.Download(url, zipFile);

            if (!File.Exists(Path.Join(dir, pbFile)))
                Utility.Compress.ExtractTGZ(zipFile, dir);

            // download sample picture
            string pic = "grace_hopper.jpg";
            Utility.Web.Download($"https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/{pic}", Path.Join(dir, pic));
        }
    }
}
