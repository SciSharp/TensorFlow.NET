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
using TensorFlowNET.Examples.Text.cnn_models;
using TensorFlowNET.Examples.TextClassification;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples.CnnTextClassification
{
    /// <summary>
    /// https://github.com/dongjun-Lee/text-classification-models-tf
    /// </summary>
    public class TextClassificationTrain : IExample
    {
        public int Priority => 100;
        public bool Enabled { get; set; } = false;
        public string Name => "Text Classification";
        public int? DataLimit = null;
        public bool ImportGraph { get; set; } = true;

        private string dataDir = "text_classification";
        private string dataFileName = "dbpedia_csv.tar.gz";

        public string model_name = "vd_cnn"; // word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn

        private const int CHAR_MAX_LEN = 1014;
        private const int WORD_MAX_LEN = 1014;
        private const int NUM_CLASS = 2;
        private const int BATCH_SIZE = 64;
        private const int NUM_EPOCHS = 10;
        protected float loss_value = 0;

        public bool Run()
        {
            PrepareData();
            var graph = tf.Graph().as_default();
            return with(tf.Session(graph), sess =>
            {
                if (ImportGraph)
                    return RunWithImportedGraph(sess, graph);
                else
                    return RunWithBuiltGraph(sess, graph);
            });
        }

        protected virtual bool RunWithImportedGraph(Session sess, Graph graph)
        {
            var stopwatch = Stopwatch.StartNew();
            Console.WriteLine("Building dataset...");
            var (x, y, alphabet_size) = DataHelpers.build_char_dataset("train", model_name, CHAR_MAX_LEN, DataLimit = null);
            Console.WriteLine("\tDONE ");

            var (train_x, valid_x, train_y, valid_y) = train_test_split(x, y, test_size: 0.15f);
            Console.WriteLine("Training set size: " + train_x.Length);
            Console.WriteLine("Test set size: " + valid_x.Length);

            Console.WriteLine("Import graph...");
            var meta_file = model_name + ".meta";
            tf.train.import_meta_graph(Path.Join("graph", meta_file));
            Console.WriteLine("\tDONE " + stopwatch.Elapsed);

            sess.run(tf.global_variables_initializer());

            var train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS);
            var num_batches_per_epoch = (len(train_x) - 1) / BATCH_SIZE + 1;
            double max_accuracy = 0;

            Tensor is_training = graph.get_operation_by_name("is_training");
            Tensor model_x = graph.get_operation_by_name("x");
            Tensor model_y = graph.get_operation_by_name("y");
            Tensor loss = graph.get_operation_by_name("loss/value");
            Tensor optimizer = graph.get_operation_by_name("loss/optimizer");
            Tensor global_step = graph.get_operation_by_name("global_step");
            Tensor accuracy = graph.get_operation_by_name("accuracy/value");
            stopwatch = Stopwatch.StartNew();
            int i = 0;
            foreach (var (x_batch, y_batch, total) in train_batches)
            {
                i++;
                var train_feed_dict = new Hashtable
                {
                    [model_x] = x_batch,
                    [model_y] = y_batch,
                    [is_training] = true,
                };
                //Console.WriteLine("x: " + x_batch.ToString() + "\n");
                //Console.WriteLine("y: " + y_batch.ToString());
                // original python:
                //_, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict = train_feed_dict)
                var result = sess.run(new ITensorOrOperation[] { optimizer, global_step, loss }, train_feed_dict);
                loss_value = result[2];
                var step = (int)result[1];
                if (step % 10 == 0 || step < 10)
                {
                    var estimate = TimeSpan.FromSeconds((stopwatch.Elapsed.TotalSeconds / i) * total);
                    Console.WriteLine($"Training on batch {i}/{total}. Estimated training time: {estimate}");
                    Console.WriteLine($"Step {step} loss: {loss_value}");
                }

                if (step % 100 == 0)
                {
                    continue;
                    // # Test accuracy with validation data for each epoch.
                    var valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1);
                    var (sum_accuracy, cnt) = (0, 0);
                    foreach (var (valid_x_batch, valid_y_batch, total_validation_batches) in valid_batches)
                    {
                        //        valid_feed_dict = {
                        //            model.x: valid_x_batch,
                        //            model.y: valid_y_batch,
                        //            model.is_training: False
                        //        }

                        //        accuracy = sess.run(model.accuracy, feed_dict = valid_feed_dict)
                        //        sum_accuracy += accuracy
                        //        cnt += 1
                    }
                    //    valid_accuracy = sum_accuracy / cnt

                    //    print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))

                    //    # Save model
                    //        if valid_accuracy > max_accuracy:
                    //        max_accuracy = valid_accuracy
                    //        saver.save(sess, "{0}/{1}.ckpt".format(args.model, args.model), global_step = step)
                    //        print("Model is saved.\n")
                }
            }

            return false;
        }

        protected virtual bool RunWithBuiltGraph(Session session, Graph graph)
        {
            Console.WriteLine("Building dataset...");
            var (x, y, alphabet_size) = DataHelpers.build_char_dataset("train", model_name, CHAR_MAX_LEN, DataLimit);

            var (train_x, valid_x, train_y, valid_y) = train_test_split(x, y, test_size: 0.15f);

            ITextClassificationModel model = null;
            switch (model_name) // word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn
            {
                case "word_cnn":
                case "char_cnn":
                case "word_rnn":
                case "att_rnn":
                case "rcnn":
                    throw new NotImplementedException();
                    break;
                case "vd_cnn":
                    model = new VdCnn(alphabet_size, CHAR_MAX_LEN, NUM_CLASS);
                    break;
            }
            // todo train the model
            return false;
        }

        // TODO: this originally is an SKLearn utility function. it randomizes train and test which we don't do here
        //private (NDArray, NDArray, NDArray, NDArray) train_test_split(NDArray x, NDArray y, float test_size = 0.3f)
        //{
        //    Console.WriteLine("Splitting in Training and Testing data...");
        //    int len = x.shape[0];
        //    //int classes = y.Data<int>().Distinct().Count();
        //    //int samples = len / classes;
        //    int train_size = (int)Math.Round(len * (1 - test_size));
        //    var train_x = x[new Slice(stop: train_size), new Slice()];
        //    var valid_x = x[new Slice(start: train_size + 1), new Slice()];
        //    var train_y = y[new Slice(stop: train_size)];
        //    var valid_y = y[new Slice(start: train_size + 1)];
        //    Console.WriteLine("\tDONE");
        //    return (train_x, valid_x, train_y, valid_y);
        //}

        private (int[][], int[][], int[], int[]) train_test_split(int[][] x, int[] y, float test_size = 0.3f)
        {
            Console.WriteLine("Splitting in Training and Testing data...");
            var stopwatch = Stopwatch.StartNew();
            int len = x.Length;
            //int classes = y.Distinct().Count();
            //int samples = len / classes;
            int train_size = int.Parse((len * (1 - test_size)).ToString());

            //var train_x = new List<int[]>();
            //var valid_x = new List<int[]>();
            //var train_y = new List<int>();
            //var valid_y = new List<int>();

            //for (int i = 0; i < classes; i++)
            //{
            //    for (int j = 0; j < samples; j++)
            //    {
            //        int idx = i * samples + j;
            //        if (idx < train_size + samples * i)
            //        {
            //            train_x.Add(x[idx]);
            //            train_y.Add(y[idx]);
            //        }
            //        else
            //        {
            //            valid_x.Add(x[idx]);
            //            valid_y.Add(y[idx]);
            //        }
            //    }
            //}
            var random = new Random(17);

            // we collect indices of labels 
            var labels = new Dictionary<int, HashSet<int>>();
            var shuffled_indices = Shuffle<int>(random, range(len).ToArray());
            foreach (var i in shuffled_indices)
            {
                var label = y[i];
                if (!labels.ContainsKey(i))
                    labels[label] = new HashSet<int>();
                labels[label].Add(i);
            }

            var train_x = new int[train_size][];
            var valid_x = new int[len - train_size][];
            var train_y = new int[train_size];
            var valid_y = new int[len - train_size];

            FillWithShuffledLabels(x, y, train_x, train_y, random, labels);
            FillWithShuffledLabels(x, y, valid_x, valid_y, random, labels);

            Console.WriteLine("\tDONE " + stopwatch.Elapsed);
            return (train_x, valid_x, train_y, valid_y);
        }

        private static void FillWithShuffledLabels(int[][] x, int[] y, int[][] shuffled_x, int[] shuffled_y, Random random, Dictionary<int, HashSet<int>> labels)
        {
            int i = 0;
            while (i < shuffled_x.Length)
            {
                foreach (var key in Shuffle<int>(random, labels.Keys.ToArray()))
                {
                    var set = labels[key];
                    var index = set.First();
                    if (set.Count == 0)
                        labels.Remove(key); // remove the set as it is empty
                    shuffled_x[i] = x[index];
                    shuffled_y[i] = y[index];
                    i++;
                }
            }
        }

        public static T[] Shuffle<T>(Random rng, T[] array)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
            return array;
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
            string url = "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz";
            Web.Download(url, dataDir, dataFileName);
            Compress.ExtractTGZ(Path.Join(dataDir, dataFileName), dataDir);

            if (ImportGraph)
            {
                // download graph meta data
                var meta_file = model_name + ".meta";
                var meta_path = Path.Combine("graph", meta_file);
                if (File.GetLastWriteTime(meta_path) < new DateTime(2019, 05, 11))
                {
                    // delete old cached file which contains errors
                    Console.WriteLine("Discarding cached file: " + meta_path);
                    File.Delete(meta_path);
                }
                url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/" + meta_file;
                Web.Download(url, "graph", meta_file);
            }
        }
    }
}
