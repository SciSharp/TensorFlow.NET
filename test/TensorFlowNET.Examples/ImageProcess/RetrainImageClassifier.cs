using NumSharp;
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
            Tensor final_tensor = graph.OperationByName("final_result");
            Tensor ground_truth_input = graph.OperationByName("input/GroundTruthInput");

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

                // Create the operations we need to evaluate the accuracy of our new layer.
                var (evaluation_step, _) = add_evaluation_step(final_tensor, ground_truth_input);

                // Merge all the summaries and write them out to the summaries_dir
                var merged = tf.summary.merge_all();
                var train_writer = tf.summary.FileWriter(summaries_dir + "/train", sess.graph);
            });

            return false;
        }

        /// <summary>
        /// Inserts the operations we need to evaluate the accuracy of our results.
        /// </summary>
        /// <param name="result_tensor"></param>
        /// <param name="ground_truth_tensor"></param>
        /// <returns></returns>
        private (Tensor, Tensor) add_evaluation_step(Tensor result_tensor, Tensor ground_truth_tensor)
        {
            Tensor evaluation_step = null, correct_prediction = null, prediction = null;

            with(tf.name_scope("accuracy"), scope =>
            {
                with(tf.name_scope("correct_prediction"), delegate
                {
                    prediction = tf.argmax(result_tensor, 1);
                    correct_prediction = tf.equal(prediction, ground_truth_tensor);
                });

                with(tf.name_scope("accuracy"), delegate
                {
                    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
                });
            });

            tf.summary.scalar("accuracy", evaluation_step);
            return (evaluation_step, prediction);
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
                        how_many_bottlenecks++;
                        if (how_many_bottlenecks % 100 == 0)
                            print($"{how_many_bottlenecks} bottleneck files created.");
                    }
                }
            }
        }

        private float[] get_or_create_bottleneck(Session sess, Dictionary<string, Dictionary<string, string[]>> image_lists, 
            string label_name, int index, string image_dir, string category, string bottleneck_dir, 
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor, Tensor resized_input_tensor,
            Tensor bottleneck_tensor, string module_name)
        {
            var label_lists = image_lists[label_name];
            var sub_dir_path = Path.Join(bottleneck_dir, label_name);
            Directory.CreateDirectory(sub_dir_path);
            string bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category, module_name);

            if (!File.Exists(bottleneck_path))
                create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                                       image_dir, category, sess, jpeg_data_tensor,
                                       decoded_image_tensor, resized_input_tensor,
                                       bottleneck_tensor);
            var bottleneck_string = File.ReadAllText(bottleneck_path);
            var bottleneck_values = Array.ConvertAll(bottleneck_string.Split(','), x => float.Parse(x));
            return bottleneck_values;
        }

        private void create_bottleneck_file(string bottleneck_path, Dictionary<string, Dictionary<string, string[]>> image_lists, 
            string label_name, int index, string image_dir, string category, Session sess, 
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor, Tensor resized_input_tensor, Tensor bottleneck_tensor)
        {
            // Create a single bottleneck file.
            print("Creating bottleneck at " + bottleneck_path);
            var image_path = get_image_path(image_lists, label_name, index, image_dir, category);
            if (!File.Exists(image_path))
                print($"File does not exist {image_path}");

            var image_data = File.ReadAllBytes(image_path);
            var bottleneck_values = run_bottleneck_on_image(
                sess, image_data, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor);
            var values = bottleneck_values.Data<float>();
            var bottleneck_string = string.Join(",", values);
            File.WriteAllText(bottleneck_path, bottleneck_string);
        }

        /// <summary>
        /// Runs inference on an image to extract the 'bottleneck' summary layer.
        /// </summary>
        /// <param name="sess">Current active TensorFlow Session.</param>
        /// <param name="image_data">Data of raw JPEG data.</param>
        /// <param name="image_data_tensor">Input data layer in the graph.</param>
        /// <param name="decoded_image_tensor">Output of initial image resizing and preprocessing.</param>
        /// <param name="resized_input_tensor">The input node of the recognition graph.</param>
        /// <param name="bottleneck_tensor">Layer before the final softmax.</param>
        /// <returns></returns>
        private NDArray run_bottleneck_on_image(Session sess, byte[] image_data, Tensor image_data_tensor,
                            Tensor decoded_image_tensor, Tensor resized_input_tensor, Tensor bottleneck_tensor)
        {
            // First decode the JPEG image, resize it, and rescale the pixel values.
            var resized_input_values = sess.run(decoded_image_tensor, new FeedItem(image_data_tensor, image_data));
            // Then run it through the recognition network.
            var bottleneck_values = sess.run(bottleneck_tensor, new FeedItem(resized_input_tensor, resized_input_values));
            bottleneck_values = np.squeeze(bottleneck_values);
            return bottleneck_values;
        }

        private string get_bottleneck_path(Dictionary<string, Dictionary<string, string[]>> image_lists, string label_name, int index, 
            string bottleneck_dir, string category, string module_name)
        {
            module_name = (module_name.Replace("://", "~")  // URL scheme.
                 .Replace('/', '~')  // URL and Unix paths.
                 .Replace(':', '~').Replace('\\', '~'));  // Windows paths.
            return get_image_path(image_lists, label_name, index, bottleneck_dir,
                                  category) + "_" + module_name + ".txt";
        }

        private string get_image_path(Dictionary<string, Dictionary<string, string[]>> image_lists, string label_name,
            int index, string image_dir, string category)
        {
            if (!image_lists.ContainsKey(label_name))
                print($"Label does not exist {label_name}");

            var label_lists = image_lists[label_name];
            if (!label_lists.ContainsKey(category))
                print($"Category does not exist {category}");
            var category_list = label_lists[category];
            if (category_list.Length == 0)
                print($"Label {label_name} has no images in the category {category}.");

            var mod_index = index % len(category_list);
            var base_name = category_list[mod_index].Split(Path.DirectorySeparatorChar).Last();
            var sub_dir = label_name;
            var full_path = Path.Join(image_dir, sub_dir, base_name);
            return full_path;
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
