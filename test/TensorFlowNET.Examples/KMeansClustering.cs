using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Clustering;
using TensorFlowNET.Examples.Utility;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Implement K-Means algorithm with TensorFlow.NET, and apply it to classify
    /// handwritten digit images.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/kmeans.py
    /// </summary>
    public class KMeansClustering : Python, IExample
    {
        public int Priority => 8;
        public bool Enabled { get; set; } = true;
        public string Name => "K-means Clustering";

        public int? train_size = null;
        public int validation_size = 5000;
        public int? test_size = null;
        public int batch_size = 1024; // The number of samples per batch

        Datasets mnist;
        NDArray full_data_x;
        int num_steps = 50; // Total steps to train
        int k = 25; // The number of clusters
        int num_classes = 10; // The 10 digits
        int num_features = 784; // Each image is 28x28 pixels

        public bool Run()
        {
            PrepareData();

            var graph = tf.Graph().as_default();

            tf.train.import_meta_graph("kmeans.meta");

            // Input images
            var X = graph.get_operation_by_name("Placeholder").output; // tf.placeholder(tf.float32, shape: new TensorShape(-1, num_features));
            //  Labels (for assigning a label to a centroid and testing)
            var Y = graph.get_operation_by_name("Placeholder_1").output; // tf.placeholder(tf.float32, shape: new TensorShape(-1, num_classes));

            // K-Means Parameters
            //var kmeans = new KMeans(X, k, distance_metric: KMeans.COSINE_DISTANCE, use_mini_batch: true);

            // Build KMeans graph
            //var training_graph = kmeans.training_graph();

            var init_vars = tf.global_variables_initializer();
            Tensor init_op = graph.get_operation_by_name("cond/Merge");
            var train_op = graph.get_operation_by_name("group_deps");
            Tensor avg_distance = graph.get_operation_by_name("Mean");
            Tensor cluster_idx = graph.get_operation_by_name("Squeeze_1");

            with(tf.Session(graph), sess =>
            {
                sess.run(init_vars, new FeedItem(X, full_data_x));
                sess.run(init_op, new FeedItem(X, full_data_x));

                // Training
                NDArray result = null;
                foreach(var i in range(1, num_steps + 1))
                {
                    result = sess.run(new ITensorOrOperation[] { train_op, avg_distance, cluster_idx }, new FeedItem(X, full_data_x));
                    if (i % 2 == 0 || i == 1)
                        print($"Step {i}, Avg Distance: {result[1]}");
                }

                var idx = result[2];

                // Assign a label to each centroid
                // Count total number of labels per centroid, using the label of each training
                // sample to their closest centroid (given by 'idx')
                var counts = np.zeros(k, num_classes);
                foreach (var i in range(idx.len))
                    counts[idx[i]] += mnist.train.labels[i];

            });

            return false;
        }

        public void PrepareData()
        {
            mnist = MnistDataSet.read_data_sets("mnist", one_hot: true, train_size: train_size, validation_size:validation_size, test_size:test_size);
            full_data_x = mnist.train.images;
        }
    }
}
