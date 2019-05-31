using Google.Protobuf;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;
using Console = Colorful.Console;

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

        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = true;

        public string Name => "Retrain Image Classifier";

        const string data_dir = "retrain_images";
        string summaries_dir = Path.Join(data_dir, "retrain_logs");
        string image_dir = Path.Join(data_dir, "flower_photos");
        string bottleneck_dir = Path.Join(data_dir, "bottleneck");
        string output_graph = Path.Join(data_dir, "output_graph.pb");
        string output_labels = Path.Join(data_dir, "output_labels.txt");
        // The location where variable checkpoints will be stored.
        string CHECKPOINT_NAME = Path.Join(data_dir, "_retrain_checkpoint");
        string tfhub_module = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3";
        string final_tensor_name = "final_result";
        float testing_percentage = 0.1f;
        float validation_percentage = 0.1f;
        float learning_rate = 0.01f;
        Tensor resized_image_tensor;
        Dictionary<string, Dictionary<string, string[]>> image_lists;
        int how_many_training_steps = 100;
        int eval_step_interval = 10;
        int train_batch_size = 100;
        int test_batch_size = -1;
        int validation_batch_size = 100;
        int intermediate_store_frequency = 0;
        int class_count = 0;
        const int MAX_NUM_IMAGES_PER_CLASS = 134217727;
        Operation train_step;
        Tensor final_tensor;
        Tensor bottleneck_input;
        Tensor cross_entropy;
        Tensor ground_truth_input;

        public bool Run()
        {
            PrepareData();

            // Set up the pre-trained graph.
            var (graph, bottleneck_tensor, resized_image_tensor, wants_quantization) =
                create_module_graph();

            // Add the new layer that we'll be training.
            with(graph.as_default(), delegate
            {
                (train_step, cross_entropy, bottleneck_input,
                 ground_truth_input, final_tensor) = add_final_retrain_ops(
                     class_count, final_tensor_name, bottleneck_tensor,
                     wants_quantization, is_training: true);
            });

            var sw = new Stopwatch();

            return with(tf.Session(graph), sess =>
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
                var validation_writer = tf.summary.FileWriter(summaries_dir + "/validation", sess.graph);

                // Create a train saver that is used to restore values into an eval graph
                // when exporting models.
                var train_saver = tf.train.Saver();
                sw.Restart();

                for (int i = 0; i < how_many_training_steps; i++)
                {
                    var (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
                         sess, image_lists, train_batch_size, "training",
                         bottleneck_dir, image_dir, jpeg_data_tensor,
                         decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                         tfhub_module);

                    // Feed the bottlenecks and ground truth into the graph, and run a training
                    // step. Capture training summaries for TensorBoard with the `merged` op.
                    var results = sess.run(
                          new ITensorOrOperation[] { merged, train_step },
                          new FeedItem(bottleneck_input, train_bottlenecks),
                          new FeedItem(ground_truth_input, train_ground_truth));
                    var train_summary = results[0];

                    // TODO
                    train_writer.add_summary(train_summary, i);

                    // Every so often, print out how well the graph is training.
                    bool is_last_step = (i + 1 == how_many_training_steps);
                    if ((i % eval_step_interval) == 0 || is_last_step)
                    {
                        results = sess.run(
                            new Tensor[] { evaluation_step, cross_entropy },
                            new FeedItem(bottleneck_input, train_bottlenecks),
                            new FeedItem(ground_truth_input, train_ground_truth));
                        (float train_accuracy, float cross_entropy_value) = (results[0], results[1]);
                        print($"{DateTime.Now}: Step {i + 1}: Train accuracy = {train_accuracy * 100}%,  Cross entropy = {cross_entropy_value.ToString("G4")}");

                        var (validation_bottlenecks, validation_ground_truth, _) = get_random_cached_bottlenecks(
                            sess, image_lists, validation_batch_size, "validation",
                            bottleneck_dir, image_dir, jpeg_data_tensor,
                            decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                            tfhub_module);

                        // Run a validation step and capture training summaries for TensorBoard
                        // with the `merged` op.
                        results = sess.run(new Tensor[] { merged, evaluation_step },
                            new FeedItem(bottleneck_input, validation_bottlenecks),
                            new FeedItem(ground_truth_input, validation_ground_truth));

                        (string validation_summary, float validation_accuracy) = (results[0], results[1]);

                        validation_writer.add_summary(validation_summary, i);
                        print($"{DateTime.Now}: Step {i + 1}: Validation accuracy = {validation_accuracy * 100}% (N={len(validation_bottlenecks)}) {sw.ElapsedMilliseconds}ms");
                        sw.Restart();
                    }

                    // Store intermediate results
                    int intermediate_frequency = intermediate_store_frequency;
                    if (intermediate_frequency > 0 && i % intermediate_frequency == 0 && i > 0)
                    {

                    }
                }

                // After training is complete, force one last save of the train checkpoint.
                train_saver.save(sess, CHECKPOINT_NAME);

                // We've completed all our training, so run a final test evaluation on
                // some new images we haven't used before.
                var (test_accuracy, predictions) = run_final_eval(sess, null, class_count, image_lists,
                               jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                               bottleneck_tensor);

                // Write out the trained graph and labels with the weights stored as
                // constants.
                print($"Save final result to : {output_graph}");
                save_graph_to_file(output_graph, class_count);
                File.WriteAllText(output_labels, string.Join("\n", image_lists.Keys));
                return test_accuracy > 0.75f;
            });
        }

        /// <summary>
        /// Runs a final evaluation on an eval graph using the test data set.
        /// </summary>
        /// <param name="train_session"></param>
        /// <param name="module_spec"></param>
        /// <param name="class_count"></param>
        /// <param name="image_lists"></param>
        /// <param name="jpeg_data_tensor"></param>
        /// <param name="decoded_image_tensor"></param>
        /// <param name="resized_image_tensor"></param>
        /// <param name="bottleneck_tensor"></param>
        private (float, NDArray) run_final_eval(Session train_session, object module_spec, int class_count, 
            Dictionary<string, Dictionary<string, string[]>> image_lists,
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor,
            Tensor resized_image_tensor, Tensor bottleneck_tensor)
        {
            var (test_bottlenecks, test_ground_truth, test_filenames) = get_random_cached_bottlenecks(train_session, image_lists,
                                    test_batch_size, "testing", bottleneck_dir, image_dir, jpeg_data_tensor,
                                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor, tfhub_module);

            var (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step,
                prediction) = build_eval_session(class_count);

            var results = eval_session.run(new Tensor[] { evaluation_step, prediction },
                  new FeedItem(bottleneck_input, test_bottlenecks),
                  new FeedItem(ground_truth_input, test_ground_truth));

            print($"final test accuracy: {((float)results[0] * 100).ToString("G4")}% (N={len(test_bottlenecks)})");

            return (results[0], results[1]);
        }

        private (Session, Tensor, Tensor, Tensor, Tensor, Tensor) 
            build_eval_session(int class_count)
        {
            // If quantized, we need to create the correct eval graph for exporting.
            var (eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization) = create_module_graph();
            var eval_sess = tf.Session(graph: eval_graph);
            Tensor evaluation_step = null;
            Tensor prediction = null;

            with(eval_graph.as_default(), graph =>
            {
                // Add the new layer for exporting.
                var (_, _, bottleneck_input, ground_truth_input, final_tensor) = 
                    add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
                        wants_quantization, is_training: false);

                // Now we need to restore the values from the training graph to the eval
                // graph.
                tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME);

                (evaluation_step, prediction) = add_evaluation_step(final_tensor,
                                                      ground_truth_input);
            });

            return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input,
                evaluation_step, prediction);
        }

        /// <summary>
        /// Adds a new softmax and fully-connected layer for training and eval.
        /// 
        /// We need to retrain the top layer to identify our new classes, so this function
        /// adds the right operations to the graph, along with some variables to hold the
        /// weights, and then sets up all the gradients for the backward pass.
        /// 
        /// The set up for the softmax and fully-connected layers is based on:
        /// https://www.tensorflow.org/tutorials/mnist/beginners/index.html
        /// </summary>
        /// <param name="class_count"></param>
        /// <param name="final_tensor_name"></param>
        /// <param name="bottleneck_tensor"></param>
        /// <param name="quantize_layer"></param>
        /// <param name="is_training"></param>
        /// <returns></returns>
        private (Operation, Tensor, Tensor, Tensor, Tensor) add_final_retrain_ops(int class_count, string final_tensor_name, 
            Tensor bottleneck_tensor, bool quantize_layer, bool is_training)
        {
            var (batch_size, bottleneck_tensor_size) = (bottleneck_tensor.GetShape().Dimensions[0], bottleneck_tensor.GetShape().Dimensions[1]);
            with(tf.name_scope("input"), scope =>
            {
                bottleneck_input = tf.placeholder_with_default(
                    bottleneck_tensor,
                    shape: bottleneck_tensor.GetShape().Dimensions,
                    name: "BottleneckInputPlaceholder");

                ground_truth_input = tf.placeholder(tf.int64, new TensorShape(batch_size), name: "GroundTruthInput");
            });

            // Organizing the following ops so they are easier to see in TensorBoard.
            string layer_name = "final_retrain_ops";
            Tensor logits = null;
            with(tf.name_scope(layer_name), scope =>
            {
                RefVariable layer_weights = null;
                with(tf.name_scope("weights"), delegate
                {
                    var initial_value = tf.truncated_normal(new int[] { bottleneck_tensor_size, class_count }, stddev: 0.001f);
                    layer_weights = tf.Variable(initial_value, name: "final_weights");
                    variable_summaries(layer_weights);
                });

                RefVariable layer_biases = null;
                with(tf.name_scope("biases"), delegate
                {
                    layer_biases = tf.Variable(tf.zeros((class_count)), name: "final_biases");
                    variable_summaries(layer_biases);
                });

                with(tf.name_scope("Wx_plus_b"), delegate
                {
                    logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases;
                    tf.summary.histogram("pre_activations", logits);
                });
            });

            final_tensor = tf.nn.softmax(logits, name: final_tensor_name);

            // The tf.contrib.quantize functions rewrite the graph in place for
            // quantization. The imported model graph has already been rewritten, so upon
            // calling these rewrites, only the newly added final layer will be
            // transformed.
            if (quantize_layer)
            {
                throw new NotImplementedException("quantize_layer");
                /*if (is_training)
                    tf.contrib.quantize.create_training_graph();
                else
                    tf.contrib.quantize.create_eval_graph();*/
            }

            tf.summary.histogram("activations", final_tensor);

            // If this is an eval graph, we don't need to add loss ops or an optimizer.
            if (!is_training)
                return (null, null, bottleneck_input, ground_truth_input, final_tensor);

            Tensor cross_entropy_mean = null;
            with(tf.name_scope("cross_entropy"), delegate
            {
                cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                    labels: ground_truth_input, logits: logits);
            });

            tf.summary.scalar("cross_entropy", cross_entropy_mean);

            with(tf.name_scope("train"), delegate
            {
                var optimizer = tf.train.GradientDescentOptimizer(learning_rate);
                train_step = optimizer.minimize(cross_entropy_mean);
            });

            return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
                final_tensor);
        }

        private void variable_summaries(RefVariable var)
        {
            with(tf.name_scope("summaries"), delegate
            {
                var mean = tf.reduce_mean(var);
                tf.summary.scalar("mean", mean);
                Tensor stddev = null;
                with(tf.name_scope("stddev"), delegate {
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)));
                });
                tf.summary.scalar("stddev", stddev);
                tf.summary.scalar("max", tf.reduce_max(var));
                tf.summary.scalar("min", tf.reduce_min(var));
                tf.summary.histogram("histogram", var);
            });
        }

        private (Graph, Tensor, Tensor, bool) create_module_graph()
        {
            var (height, width) = (299, 299);
            
            return with(tf.Graph().as_default(), graph =>
            {
                tf.train.import_meta_graph("graph/InceptionV3.meta");
                Tensor resized_input_tensor = graph.OperationByName("Placeholder"); //tf.placeholder(tf.float32, new TensorShape(-1, height, width, 3));
                // var m = hub.Module(module_spec);
                Tensor bottleneck_tensor = graph.OperationByName("module_apply_default/hub_output/feature_vector/SpatialSqueeze");// m(resized_input_tensor);
                var wants_quantization = false;
                return (graph, bottleneck_tensor, resized_input_tensor, wants_quantization);
            });
        }

        private (NDArray, long[], string[]) get_random_cached_bottlenecks(Session sess, Dictionary<string, Dictionary<string, string[]>> image_lists, 
            int how_many, string category, string bottleneck_dir, string image_dir, 
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor, Tensor resized_input_tensor,
            Tensor bottleneck_tensor, string module_name)
        {
            var bottlenecks = new List<float[]>();
            var ground_truths = new List<long>();
            var filenames = new List<string>();
            class_count = image_lists.Keys.Count;
            if (how_many >= 0)
            {
                // Retrieve a random sample of bottlenecks.
                foreach (var unused_i in range(how_many))
                {
                    int label_index = new Random().Next(class_count);
                    string label_name = image_lists.Keys.ToArray()[label_index];
                    int image_index = new Random().Next(MAX_NUM_IMAGES_PER_CLASS);
                    string image_name = get_image_path(image_lists, label_name, image_index,
                                      image_dir, category);
                    var bottleneck = get_or_create_bottleneck(
                      sess, image_lists, label_name, image_index, image_dir, category,
                      bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, module_name);
                    bottlenecks.Add(bottleneck);
                    ground_truths.Add(label_index);
                    filenames.Add(image_name);
                }
            }
            else
            {
                // Retrieve all bottlenecks.
                foreach (var (label_index, label_name) in enumerate(image_lists.Keys.ToArray()))
                {
                    foreach(var (image_index, image_name) in enumerate(image_lists[label_name][category]))
                    {
                        var bottleneck = get_or_create_bottleneck(
                            sess, image_lists, label_name, image_index, image_dir, category,
                            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                            resized_input_tensor, bottleneck_tensor, module_name);

                        bottlenecks.Add(bottleneck);
                        ground_truths.Add(label_index);
                        filenames.Add(image_name);
                    }
                }
            }

            return (bottlenecks.ToArray(), ground_truths.ToArray(), filenames.ToArray());
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
                        if (how_many_bottlenecks % 300 == 0)
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

        /// <summary>
        /// Saves an graph to file, creating a valid quantized one if necessary.
        /// </summary>
        /// <param name="graph_file_name"></param>
        /// <param name="class_count"></param>
        private void save_graph_to_file(string graph_file_name, int class_count)
        {
            var (sess, _, _, _, _, _) = build_eval_session(class_count);
            var graph = sess.graph;
            var output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), new string[] { final_tensor_name });
            File.WriteAllBytes(graph_file_name, output_graph_def.ToByteArray());
        }

        public void PrepareData()
        {
            // get a set of images to teach the network about the new classes
            string fileName = "flower_photos.tgz";
            string url = $"http://download.tensorflow.org/example_images/{fileName}";
            Web.Download(url, data_dir, fileName);
            Compress.ExtractTGZ(Path.Join(data_dir, fileName), data_dir);

            // download graph meta data
            url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/InceptionV3.meta";
            Web.Download(url, "graph", "InceptionV3.meta");

            // download variables.data checkpoint file.
            url = "https://github.com/SciSharp/TensorFlow.NET/raw/master/data/tfhub_modules.zip";
            Web.Download(url, data_dir, "tfhub_modules.zip");
            Compress.UnZip(Path.Join(data_dir, "tfhub_modules.zip"), Path.Join(Path.GetTempPath(), "tfhub_modules"));

            // Prepare necessary directories that can be used during training
            Directory.CreateDirectory(summaries_dir);
            Directory.CreateDirectory(bottleneck_dir);

            // Look at the folder structure, and create lists of all the images.
            image_lists = create_image_lists();
            class_count = len(image_lists);
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
