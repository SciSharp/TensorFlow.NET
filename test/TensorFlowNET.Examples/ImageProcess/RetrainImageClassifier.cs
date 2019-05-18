using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples.ImageProcess
{
    /// <summary>
    /// In this tutorial, we will reuse the feature extraction capabilities from powerful image classifiers trained on ImageNet 
    /// and simply train a new classification layer on top. Transfer learning is a technique that shortcuts much of this 
    /// by taking a piece of a model that has already been trained on a related task and reusing it in a new model.
    /// 
    /// https://www.tensorflow.org/hub/tutorials/image_retraining
    /// </summary>
    public class RetrainImageClassifier : IExample
    {
        public int Priority => 16;

        public bool Enabled { get; set; } = false;
        public bool ImportGraph { get; set; } = true;

        public string Name => "Retrain Image Classifier";

        const string data_dir = "retrain_images";
        string summaries_dir = Path.Join(data_dir, "retrain_logs");
        string image_dir = Path.Join(data_dir, "flower_photos");
        string bottleneck_dir = Path.Join(data_dir, "bottleneck");
        string tfhub_module = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3";
        float testing_percentage = 0.1f;
        float validation_percentage = 0.1f;
        Tensor resized_image_tensor;
        Dictionary<string, Dictionary<string, string[]>> image_lists;

        public bool Run()
        {
            PrepareData();

            var graph = tf.Graph().as_default();
            tf.train.import_meta_graph("graph/InceptionV3.meta");
            Tensor bottleneck_tensor = graph.OperationByName("module_apply_default/hub_output/feature_vector/SpatialSqueeze");
            Tensor resized_image_tensor = graph.OperationByName("Placeholder");

            var sw = new Stopwatch();

            with(tf.Session(graph), sess =>
            {
                // Initialize all weights: for the module to their pretrained values,
                // and for the newly added retraining layer to random initial values.
                var init = tf.global_variables_initializer();
                sess.run(init);

                var (jpeg_data_tensor, decoded_image_tensor) = add_jpeg_decoding();

                // We'll make sure we've calculated the 'bottleneck' image summaries and
                // cached them on disk.
                cache_bottlenecks(sess, image_lists, image_dir,
                        bottleneck_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor,
                        bottleneck_tensor, tfhub_module);
            });

            return false;
        }

        /// <summary>
        /// Ensures all the training, testing, and validation bottlenecks are cached.
        /// </summary>
        /// <param name="sess"></param>
        /// <param name="image_lists"></param>
        /// <param name="image_dir"></param>
        /// <param name="bottleneck_dir"></param>
        /// <param name="jpeg_data_tensor"></param>
        /// <param name="decoded_image_tensor"></param>
        /// <param name="resized_image_tensor"></param>
        /// <param name="bottleneck_tensor"></param>
        /// <param name="tfhub_module"></param>
        private void cache_bottlenecks(Session sess, Dictionary<string, Dictionary<string, string[]>> image_lists, 
            string image_dir, string bottleneck_dir, Tensor jpeg_data_tensor, Tensor decoded_image_tensor,
            Tensor resized_input_tensor, Tensor bottleneck_tensor, string module_name)
        {
            int how_many_bottlenecks = 0;
            foreach(var (label_name, label_lists) in image_lists)
            {
                foreach(var category in new string[] { "training", "testing", "validation" })
                {
                    var category_list = label_lists[category];
                    foreach(var (index, unused_base_name) in enumerate(category_list))
                    {
                        get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category,
                            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                            resized_input_tensor, bottleneck_tensor, module_name);
                    }
                }
            }
        }

        private void get_or_create_bottleneck(Session sess, Dictionary<string, Dictionary<string, string[]>> image_lists, 
            string label_name, int index, string image_dir, string category, string bottleneck_dir, 
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor, Tensor resized_input_tensor,
            Tensor bottleneck_tensor, string module_name)
        {
            var label_lists = image_lists[label_name];
            var sub_dir_path = Path.Join(image_dir, label_name);
        }

        public void PrepareData()
        {
            // get a set of images to teach the network about the new classes
            string fileName = "flower_photos.tgz";
            string url = $"http://download.tensorflow.org/models/{fileName}";
            Web.Download(url, data_dir, fileName);
            Compress.ExtractTGZ(Path.Join(data_dir, fileName), data_dir);

            // download graph meta data
            url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/InceptionV3.meta";
            Web.Download(url, "graph", "InceptionV3.meta");

            // Prepare necessary directories that can be used during training
            Directory.CreateDirectory(summaries_dir);
            Directory.CreateDirectory(bottleneck_dir);

            // Look at the folder structure, and create lists of all the images.
            image_lists = create_image_lists();
            var class_count = len(image_lists);
            if (class_count == 0)
                print($"No valid folders of images found at {image_dir}");
            if (class_count == 1)
                print("Only one valid folder of images found at " +
                     image_dir +
                     " - multiple classes are needed for classification.");
        }

        private (Tensor, Tensor) add_jpeg_decoding()
        {
            // height, width, depth
            var input_dim = (299, 299, 3);
            var jpeg_data = tf.placeholder(tf.chars, name: "DecodeJPGInput");
            var decoded_image = tf.image.decode_jpeg(jpeg_data, channels: input_dim.Item3);
            // Convert from full range of uint8 to range [0,1] of float32.
            var decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32);
            var decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0);
            var resize_shape = tf.stack(new int[] { input_dim.Item1, input_dim.Item2 });
            var resize_shape_as_int = tf.cast(resize_shape, dtype: tf.int32);
            var resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int);
            return (jpeg_data, resized_image);
        }

        /// <summary>
        /// Builds a list of training images from the file system.
        /// </summary>
        private Dictionary<string, Dictionary<string, string[]>> create_image_lists()
        {
            var sub_dirs = tf.gfile.Walk(image_dir)
                .Select(x => x.Item1)
                .OrderBy(x => x)
                .ToArray();

            var result = new Dictionary<string, Dictionary<string, string[]>>();

            foreach(var sub_dir in sub_dirs)
            {
                var dir_name = sub_dir.Split(Path.DirectorySeparatorChar).Last();
                print($"Looking for images in '{dir_name}'");
                var file_list = Directory.GetFiles(sub_dir);
                if (len(file_list) < 20)
                    print($"WARNING: Folder has less than 20 images, which may cause issues.");

                var label_name = dir_name.ToLower();
                result[label_name] = new Dictionary<string, string[]>();
                int testing_count = (int)Math.Floor(file_list.Length * testing_percentage);
                int validation_count = (int)Math.Floor(file_list.Length * validation_percentage);
                result[label_name]["testing"] = file_list.Take(testing_count).ToArray();
                result[label_name]["validation"] = file_list.Skip(testing_count).Take(validation_count).ToArray();
                result[label_name]["training"] = file_list.Skip(testing_count + validation_count).ToArray();
            }

            return result;
        }
    }
}
