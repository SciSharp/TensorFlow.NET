using System;
using System.Collections;
using System.Collections.Generic;
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
            Console.WriteLine("Building dataset...");
            var (x, y, alphabet_size) = DataHelpers.build_char_dataset("train", model_name, CHAR_MAX_LEN, DataLimit);
            Console.WriteLine("\tDONE");

            var (train_x, valid_x, train_y, valid_y) = train_test_split(x, y, test_size: 0.15f);

            Console.WriteLine("Import graph...");
            var meta_file = model_name + ".meta";
            tf.train.import_meta_graph(Path.Join("graph", meta_file));
            Console.WriteLine("\tDONE");
            //sess.run(tf.global_variables_initializer()); // not necessary here, has already been done before meta graph export

            var train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS);
            var num_batches_per_epoch = (len(train_x) - 1); // BATCH_SIZE + 1
            double max_accuracy = 0;

            Tensor is_training = graph.get_operation_by_name("is_training");
            Tensor model_x = graph.get_operation_by_name("x");
            Tensor model_y = graph.get_operation_by_name("y");
            Tensor loss = graph.get_operation_by_name("loss/loss");
            //var optimizer_nodes = graph._nodes_by_name.Keys.Where(key => key.Contains("optimizer")).ToArray();
            Tensor optimizer = graph.get_operation_by_name("loss/optimizer");
            Tensor global_step = graph.get_operation_by_name("global_step");
            Tensor accuracy = graph.get_operation_by_name("accuracy/accuracy");

            int i = 0;
            foreach (var (x_batch, y_batch) in train_batches)
            {
                i++;
                Console.WriteLine("Training on batch " + i);
                var train_feed_dict = new Hashtable
                {
                    [model_x] = x_batch,
                    [model_y] = y_batch,
                    [is_training] = true,
                };
                // original python:
                //_, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict = train_feed_dict)
                var result = sess.run(new ITensorOrOperation[] { optimizer, global_step, loss }, train_feed_dict);
                loss_value = result[2];
                var step = result[1];
                if (step % 100 == 0)
                    Console.WriteLine($"Step {step} loss: {loss_value}");
                if (step % 2000 == 0)
                {
                    continue;
                    // # Test accuracy with validation data for each epoch.
                    var valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1);
                    var (sum_accuracy, cnt) = (0, 0);
                    foreach (var (valid_x_batch, valid_y_batch) in valid_batches)
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
        private (NDArray, NDArray, NDArray, NDArray) train_test_split(NDArray x, NDArray y, float test_size = 0.3f)
        {
            Console.WriteLine("Splitting in Training and Testing data...");
            int len = x.shape[0];
            //int classes = y.Data<int>().Distinct().Count();
            //int samples = len / classes;
            int train_size = (int)Math.Round(len * (1 - test_size));
            var train_x = x[new Slice(stop:train_size), new Slice()];
            var valid_x = x[new Slice(start: train_size+1), new Slice()];
            var train_y = y[new Slice(stop: train_size)];
            var valid_y = y[new Slice(start: train_size + 1)];
            Console.WriteLine("\tDONE");
            return (train_x, valid_x, train_y, valid_y);
        }

        private IEnumerable<(NDArray, NDArray)> batch_iter(NDArray inputs, NDArray outputs, int batch_size, int num_epochs)
        {
            var num_batches_per_epoch = (len(inputs) - 1); // batch_size + 1
            foreach (var epoch in range(num_epochs))
            {
                foreach (var batch_num in range(num_batches_per_epoch))
                {
                    var start_index = batch_num * batch_size;
                    var end_index = Math.Min((batch_num + 1) * batch_size, len(inputs));
                    yield return (inputs[new Slice(start_index, end_index)], outputs[new Slice(start_index,end_index)]);
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
                url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/" + meta_file;
                Web.Download(url, "graph", meta_file);
            }
        }
    }
}
