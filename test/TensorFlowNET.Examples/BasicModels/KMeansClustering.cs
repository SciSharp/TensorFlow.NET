using NumSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Clustering;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Implement K-Means algorithm with TensorFlow.NET, and apply it to classify
    /// handwritten digit images.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/kmeans.py
    /// </summary>
    public class KMeansClustering : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "K-means Clustering";
        public bool IsImportingGraph { get; set; } = true;

        public int? train_size = null;
        public int validation_size = 5000;
        public int? test_size = null;
        public int batch_size = 1024; // The number of samples per batch

        Datasets mnist;
        NDArray full_data_x;
        int num_steps = 20; // Total steps to train
        int k = 25; // The number of clusters
        int num_classes = 10; // The 10 digits
        int num_features = 784; // Each image is 28x28 pixels

        public bool Run()
        {
            PrepareData();

            var graph = tf.Graph().as_default();

            tf.train.import_meta_graph("graph/kmeans.meta");

            // Input images
            Tensor X = graph.get_operation_by_name("Placeholder"); // tf.placeholder(tf.float32, shape: new TensorShape(-1, num_features));
            //  Labels (for assigning a label to a centroid and testing)
            Tensor Y = graph.get_operation_by_name("Placeholder_1"); // tf.placeholder(tf.float32, shape: new TensorShape(-1, num_classes));

            // K-Means Parameters
            //var kmeans = new KMeans(X, k, distance_metric: KMeans.COSINE_DISTANCE, use_mini_batch: true);

            // Build KMeans graph
            //var training_graph = kmeans.training_graph();

            var init_vars = tf.global_variables_initializer();
            Tensor init_op = graph.get_operation_by_name("cond/Merge");
            var train_op = graph.get_operation_by_name("group_deps");
            Tensor avg_distance = graph.get_operation_by_name("Mean");
            Tensor cluster_idx = graph.get_operation_by_name("Squeeze_1");
            NDArray result = null;

            with(tf.Session(graph), sess =>
            {
                sess.run(init_vars, new FeedItem(X, full_data_x));
                sess.run(init_op, new FeedItem(X, full_data_x));

                // Training
                var sw = new Stopwatch();

                foreach (var i in range(1, num_steps + 1))
                {
                    sw.Restart();
                    result = sess.run(new ITensorOrOperation[] { train_op, avg_distance, cluster_idx }, new FeedItem(X, full_data_x));
                    sw.Stop();

                    if (i % 4 == 0 || i == 1)
                        print($"Step {i}, Avg Distance: {result[1]} Elapse: {sw.ElapsedMilliseconds}ms");
                }

                var idx = result[2].Data<int>();

                // Assign a label to each centroid
                // Count total number of labels per centroid, using the label of each training
                // sample to their closest centroid (given by 'idx')
                var counts = np.zeros((k, num_classes), np.float32);

                sw.Start();
                foreach (var i in range(idx.Length))
                {
                    var x = mnist.train.labels[i];
                    counts[idx[i]] += x;
                }
                    
                sw.Stop();
                print($"Assign a label to each centroid took {sw.ElapsedMilliseconds}ms");

                // Assign the most frequent label to the centroid
                var labels_map_array = np.argmax(counts, 1);
                var labels_map = tf.convert_to_tensor(labels_map_array);

                // Evaluation ops
                // Lookup: centroid_id -> label
                var cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx);

                // Compute accuracy
                var correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32));
                var cast = tf.cast(correct_prediction, tf.float32);
                var accuracy_op = tf.reduce_mean(cast);

                // Test Model
                var (test_x, test_y) = (mnist.test.images, mnist.test.labels);
                result = sess.run(accuracy_op, new FeedItem(X, test_x), new FeedItem(Y, test_y));
                print($"Test Accuracy: {result}");
            });

            return (float)result > 0.70;
        }

        public void PrepareData()
        {
            mnist = MnistDataSet.read_data_sets("mnist", one_hot: true, train_size: train_size, validation_size:validation_size, test_size:test_size);
            full_data_x = mnist.train.images;

            // download graph meta data
            string url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/kmeans.meta";
            Web.Download(url, "graph", "kmeans.meta");
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
