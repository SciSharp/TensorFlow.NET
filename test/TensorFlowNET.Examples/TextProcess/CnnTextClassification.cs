using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Sessions;
using TensorFlowNET.Examples.Text.cnn_models;
using TensorFlowNET.Examples.TextClassification;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://github.com/dongjun-Lee/text-classification-models-tf
    /// </summary>
    public class CnnTextClassification : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "CNN Text Classification";
        public int? DataLimit = null;
        public bool IsImportingGraph { get; set; } = false;

        private const string dataDir = "word_cnn";
        private string dataFileName = "dbpedia_csv.tar.gz";

        private const string TRAIN_PATH = "word_cnn/dbpedia_csv/train.csv";
        private const string TEST_PATH = "word_cnn/dbpedia_csv/test.csv";
        
        private const int NUM_CLASS = 14;
        private const int BATCH_SIZE = 64;
        private const int NUM_EPOCHS = 10;
        private const int WORD_MAX_LEN = 100;
        private const int CHAR_MAX_LEN = 1014;
        
        protected float loss_value = 0;
        int vocabulary_size = 50000;
        NDArray train_x, valid_x, train_y, valid_y;

        public bool Run()
        {
            PrepareData();

            Train();

            return true;
        }

        // TODO: this originally is an SKLearn utility function. it randomizes train and test which we don't do here
        private (NDArray, NDArray, NDArray, NDArray) train_test_split(NDArray x, NDArray y, float test_size = 0.3f)
        {
            Console.WriteLine("Splitting in Training and Testing data...");
            int len = x.shape[0];
            //int classes = y.Data<int>().Distinct().Count();
            //int samples = len / classes;
            int train_size = (int)Math.Round(len * (1 - test_size));
            var train_x = x[new Slice(stop: train_size), new Slice()];
            var valid_x = x[new Slice(start: train_size), new Slice()];
            var train_y = y[new Slice(stop: train_size)];
            var valid_y = y[new Slice(start: train_size)];
            Console.WriteLine("\tDONE");
            return (train_x, valid_x, train_y, valid_y);
        }

        private static void FillWithShuffledLabels(int[][] x, int[] y, int[][] shuffled_x, int[] shuffled_y, Random random, Dictionary<int, HashSet<int>> labels)
        {
            int i = 0;
            var label_keys = labels.Keys.ToArray();
            while (i < shuffled_x.Length)
            {
                var key = label_keys[random.Next(label_keys.Length)];
                var set = labels[key];
                var index = set.First();
                if (set.Count == 0)
                {
                    labels.Remove(key); // remove the set as it is empty
                    label_keys = labels.Keys.ToArray();
                }
                shuffled_x[i] = x[index];
                shuffled_y[i] = y[index];
                i++;
            }
        }

        private IEnumerable<(NDArray, NDArray, int)> batch_iter(NDArray inputs, NDArray outputs, int batch_size, int num_epochs)
        {
            var num_batches_per_epoch = (len(inputs) - 1) / batch_size + 1;
            var total_batches = num_batches_per_epoch * num_epochs;
            foreach (var epoch in range(num_epochs))
            {
                foreach (var batch_num in range(num_batches_per_epoch))
                {
                    var start_index = batch_num * batch_size;
                    var end_index = Math.Min((batch_num + 1) * batch_size, len(inputs));
                    if (end_index <= start_index)
                        break;
                    yield return (inputs[new Slice(start_index, end_index)], outputs[new Slice(start_index, end_index)], total_batches);
                }
            }
        }

        public void PrepareData()
        {
            // full dataset https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz
            var url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/data/dbpedia_subset.zip";
            Web.Download(url, dataDir, "dbpedia_subset.zip");
            Compress.UnZip(Path.Combine(dataDir, "dbpedia_subset.zip"), Path.Combine(dataDir, "dbpedia_csv"));

            Console.WriteLine("Building dataset...");

            int alphabet_size = 0;

            var word_dict = DataHelpers.build_word_dict(TRAIN_PATH);
            vocabulary_size = len(word_dict);
            var (x, y) = DataHelpers.build_word_dataset(TRAIN_PATH, word_dict, WORD_MAX_LEN);

            Console.WriteLine("\tDONE ");

            var (train_x, valid_x, train_y, valid_y) = train_test_split(x, y, test_size: 0.15f);
            Console.WriteLine("Training set size: " + train_x.len);
            Console.WriteLine("Test set size: " + valid_x.len);
        }

        public Graph ImportGraph()
        {
            var graph = tf.Graph().as_default();

            // download graph meta data
            var meta_file = "word_cnn.meta";
            var meta_path = Path.Combine("graph", meta_file);
            if (File.GetLastWriteTime(meta_path) < new DateTime(2019, 05, 11))
            {
                // delete old cached file which contains errors
                Console.WriteLine("Discarding cached file: " + meta_path);
                File.Delete(meta_path);
            }
            var url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/" + meta_file;
            Web.Download(url, "graph", meta_file);

            Console.WriteLine("Import graph...");
            tf.train.import_meta_graph(Path.Join("graph", meta_file));
            Console.WriteLine("\tDONE ");

            return graph;
        }

        public Graph BuildGraph()
        {
            var graph = tf.Graph().as_default();

            var embedding_size = 128;
            var learning_rate = 0.001f;
            var filter_sizes = new int[3, 4, 5];
            var num_filters = 100;
            var document_max_len = 100;

            var x = tf.placeholder(tf.int32, new TensorShape(-1, document_max_len), name: "x");
            var y = tf.placeholder(tf.int32, new TensorShape(-1), name: "y");
            var is_training = tf.placeholder(tf.@bool, new TensorShape(), name: "is_training");
            var global_step = tf.Variable(0, trainable: false);
            var keep_prob = tf.where(is_training, 0.5f, 1.0f);
            Tensor x_emb = null;

            with(tf.name_scope("embedding"), scope =>
            {
                var init_embeddings = tf.random_uniform(new int[] { vocabulary_size, embedding_size });
                var embeddings = tf.get_variable("embeddings", initializer: init_embeddings);
                x_emb = tf.nn.embedding_lookup(embeddings, x);
                x_emb = tf.expand_dims(x_emb, -1);
            });

            var pooled_outputs = new List<Tensor>();
            for (int len = 0; len < filter_sizes.Rank; len++)
            {
                int filter_size = filter_sizes.GetLength(len);
                var conv = tf.layers.conv2d(
                    x_emb,
                    filters: num_filters,
                    kernel_size: new int[] { filter_size, embedding_size },
                    strides: new int[] { 1, 1 },
                    padding: "VALID",
                    activation: tf.nn.relu());

                var pool = tf.layers.max_pooling2d(
                    conv,
                    pool_size: new[] { document_max_len - filter_size + 1, 1 },
                    strides: new[] { 1, 1 },
                    padding: "VALID");

                pooled_outputs.Add(pool);
            }

            var h_pool = tf.concat(pooled_outputs, 3);
            var h_pool_flat = tf.reshape(h_pool, new TensorShape(-1, num_filters * filter_sizes.Rank));

            with(tf.name_scope("dropout"), delegate
            {
                var h_drop = tf.nn.dropout(h_pool_flat, keep_prob);
            });

            Tensor logits = null;
            Tensor predictions = null;
            with(tf.name_scope("output"), delegate
            {
                logits = tf.layers.dense(h_pool_flat, NUM_CLASS);
                predictions = tf.argmax(logits, -1, output_type: tf.int32);
            });

            with(tf.name_scope("loss"), delegate
            {
                var sscel = tf.nn.sparse_softmax_cross_entropy_with_logits(logits: logits, labels: y);
                var loss = tf.reduce_mean(sscel);
                var adam = tf.train.AdamOptimizer(learning_rate);
                var optimizer = adam.minimize(loss, global_step: global_step);
            });

            with(tf.name_scope("accuracy"), delegate
            {
                var correct_predictions = tf.equal(predictions, y);
                var accuracy = tf.reduce_mean(tf.cast(correct_predictions, TF_DataType.TF_FLOAT), name: "accuracy");
            });

            return graph;
        }

        private bool Train(Session sess, Graph graph)
        {
            var stopwatch = Stopwatch.StartNew();

            sess.run(tf.global_variables_initializer());
            var saver = tf.train.Saver(tf.global_variables());

            var train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS);
            var num_batches_per_epoch = (len(train_x) - 1) / BATCH_SIZE + 1;
            double max_accuracy = 0;

            Tensor is_training = graph.OperationByName("is_training");
            Tensor model_x = graph.OperationByName("x");
            Tensor model_y = graph.OperationByName("y");
            Tensor loss = graph.OperationByName("loss/Mean");
            Operation optimizer = graph.OperationByName("loss/Adam");
            Tensor global_step = graph.OperationByName("Variable");
            Tensor accuracy = graph.OperationByName("accuracy/accuracy");
            stopwatch = Stopwatch.StartNew();
            int i = 0;
            foreach (var (x_batch, y_batch, total) in train_batches)
            {
                i++;
                var train_feed_dict = new FeedDict
                {
                    [model_x] = x_batch,
                    [model_y] = y_batch,
                    [is_training] = true,
                };

                var result = sess.run(new ITensorOrOperation[] { optimizer, global_step, loss }, train_feed_dict);
                loss_value = result[2];
                var step = (int)result[1];
                if (step % 10 == 0)
                {
                    var estimate = TimeSpan.FromSeconds((stopwatch.Elapsed.TotalSeconds / i) * total);
                    Console.WriteLine($"Training on batch {i}/{total} loss: {loss_value}. Estimated training time: {estimate}");
                }

                if (step % 100 == 0)
                {
                    // Test accuracy with validation data for each epoch.
                    var valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1);
                    var (sum_accuracy, cnt) = (0.0f, 0);
                    foreach (var (valid_x_batch, valid_y_batch, total_validation_batches) in valid_batches)
                    {
                        var valid_feed_dict = new FeedDict
                        {
                            [model_x] = valid_x_batch,
                            [model_y] = valid_y_batch,
                            [is_training] = false
                        };
                        var result1 = sess.run(accuracy, valid_feed_dict);
                        float accuracy_value = result1;
                        sum_accuracy += accuracy_value;
                        cnt += 1;
                    }

                    var valid_accuracy = sum_accuracy / cnt;

                    print($"\nValidation Accuracy = {valid_accuracy}\n");

                    // Save model
                    if (valid_accuracy > max_accuracy)
                    {
                        max_accuracy = valid_accuracy;
                        saver.save(sess, $"{dataDir}/word_cnn.ckpt", global_step: step);
                        print("Model is saved.\n");
                    }
                }
            }

            return false;
        }

        public bool Train()
        {
            var graph = IsImportingGraph ? ImportGraph() : BuildGraph();

            return with(tf.Session(graph), sess => Train(sess, graph));
        }

        public bool Predict()
        {
            throw new NotImplementedException();
        }
    }
}
