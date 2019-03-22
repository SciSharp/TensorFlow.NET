using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using NumSharp.Core;
using System.Linq;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://github.com/nicolov/naive_bayes_tensorflow
    /// </summary>
    public class NaiveBayesClassifier : Python, IExample
    { 
        public Normal dist { get; set; }
        public void Run()
        {
            np.array(1.0f, 1.0f);
            var X = np.array(new float[][] { new float[] { 1.0f, 1.0f }, new float[] { 2.0f, 2.0f }, new float[] { -1.0f, -1.0f }, new float[] { -2.0f, -2.0f }, new float[] { 1.0f, -1.0f }, new float[] { 2.0f, -2.0f }, });
            var y = np.array(0,0,1,1,2,2);
            fit(X, y);
            // Create a regular grid and classify each point 
        }

        public void fit(NDArray X, NDArray y)
        {
            NDArray unique_y = y.unique<long>();

            Dictionary<long, List<List<float>>> dic = new Dictionary<long, List<List<float>>>();
            // Init uy in dic
            foreach (int uy in unique_y.Data<int>())
            {
                dic.Add(uy, new List<List<float>>());
            }
            // Separate training points by class 
            // Shape : nb_classes * nb_samples * nb_features
            int maxCount = 0;
            for (int i = 0; i < y.size; i++)
            {
                long curClass = (long)y[i];
                List<List<float>> l = dic[curClass];
                List<float> pair = new List<float>();
                pair.Add((float)X[i,0]);
                pair.Add((float)X[i, 1]);
                l.Add(pair);
                if (l.Count > maxCount)
                {
                    maxCount = l.Count;
                }
                dic[curClass] = l;
            }
            float[,,] points = new float[dic.Count, maxCount, X.shape[1]];
            foreach (KeyValuePair<long, List<List<float>>> kv in dic)
            {
                int j = (int) kv.Key;
                for (int i = 0; i < maxCount; i++)
                {
                    for (int k = 0; k < X.shape[1]; k++)
                    {
                        points[j, i, k] = kv.Value[i][k];
                    }
                }

            }
            NDArray points_by_class = np.array<float>(points);
            // estimate mean and variance for each class / feature
            // shape : nb_classes * nb_features
            var cons = tf.constant(points_by_class);
            var tup = tf.nn.moments(cons, new int[]{1});
            var mean = tup.Item1;
            var variance = tup.Item2;

            // Create a 3x2 univariate normal distribution with the 
            // Known mean and variance           
            var dist = tf.distributions.Normal(mean, tf.sqrt(variance));
            this.dist = dist;
        }

        public Tensor predict (NDArray X)
        {
            if (dist == null)
            {
                throw new ArgumentNullException("cant not find the model (normal distribution)!");
            }
            int nb_classes = (int) dist.scale().shape[0];
            int nb_features = (int)dist.scale().shape[1];

            // Conditional probabilities log P(x|c) with shape
            // (nb_samples, nb_classes)
            Tensor tile = tf.tile(new Tensor(X), new Tensor(new int[] { -1, nb_classes, nb_features }));
            Tensor r = tf.reshape(tile, new Tensor(new int[] { -1, nb_classes, nb_features }));
            var cond_probs = tf.reduce_sum(dist.log_prob(r));
            // uniform priors
            var priors = np.log(np.array<double>((1.0 / nb_classes) * nb_classes));

            // posterior log probability, log P(c) + log P(x|c)
            var joint_likelihood = tf.add(new Tensor(priors), cond_probs);
            // normalize to get (log)-probabilities

            var norm_factor = tf.reduce_logsumexp(joint_likelihood, new int[] { 1 }, true);
            var log_prob = joint_likelihood - norm_factor;
            // exp to get the actual probabilities
            return tf.exp(log_prob);
        }

        public void PrepareData()
        {
            
        }
    }
}
