using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using NumSharp;
using System.Linq;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://github.com/nicolov/naive_bayes_tensorflow
    /// </summary>
    public class NaiveBayesClassifier : Python, IExample
    {
        public int Priority => 6;
        public bool Enabled => true;
        public string Name => "Naive Bayes Classifier";

        public Normal dist { get; set; }
        public bool Run()
        {
            var X = np.array(new float[][] { new float[] { 5.1f, 3.5f},new float[] { 4.9f, 3.0f },new float[] { 4.7f, 3.2f },
                            new float[] { 4.6f, 3.1f },new float[] { 5.0f, 3.6f },new float[] { 5.4f, 3.9f },
                            new float[] { 4.6f, 3.4f },new float[] { 5.0f, 3.4f },new float[] { 4.4f, 2.9f },
                            new float[] { 4.9f, 3.1f },new float[] { 5.4f, 3.7f },new float[] {4.8f, 3.4f },
                            new float[] {4.8f, 3.0f },new float[] {4.3f, 3.0f },new float[] {5.8f, 4.0f },
                            new float[] {5.7f, 4.4f },new float[] {5.4f, 3.9f },new float[] {5.1f, 3.5f },
                            new float[] {5.7f, 3.8f },new float[] {5.1f, 3.8f },new float[] {5.4f, 3.4f },
                            new float[] {5.1f, 3.7f },new float[] {5.1f, 3.3f },new float[] {4.8f, 3.4f },
                            new float[] {5.0f, 3.0f },new float[] {5.0f , 3.4f },new float[] {5.2f, 3.5f },
                            new float[] {5.2f, 3.4f },new float[] {4.7f, 3.2f },new float[] {4.8f, 3.1f },
                            new float[] {5.4f, 3.4f },new float[] {5.2f, 4.1f},new float[] {5.5f, 4.2f },
                            new float[] {4.9f, 3.1f },new float[] {5.0f , 3.2f },new float[] {5.5f, 3.5f },
                            new float[] {4.9f, 3.6f },new float[] {4.4f, 3.0f },new float[] {5.1f, 3.4f },
                            new float[] {5.0f , 3.5f },new float[] {4.5f, 2.3f },new float[] {4.4f, 3.2f },
                            new float[] {5.0f , 3.5f },new float[] {5.1f, 3.8f },new float[] {4.8f, 3.0f},
                            new float[] {5.1f, 3.8f },new float[] {4.6f, 3.2f },new float[] { 5.3f, 3.7f },
                            new float[] {5.0f , 3.3f },new float[] {7.0f , 3.2f },new float[] {6.4f, 3.2f },
                            new float[] {6.9f, 3.1f },new float[] {5.5f, 2.3f },new float[] {6.5f, 2.8f },
                            new float[] {5.7f, 2.8f },new float[] {6.3f, 3.3f },new float[] {4.9f, 2.4f },
                            new float[] {6.6f, 2.9f },new float[] {5.2f, 2.7f },new float[] {5.0f , 2.0f },
                            new float[] {5.9f, 3.0f },new float[] {6.0f , 2.2f },new float[] {6.1f, 2.9f },
                            new float[] {5.6f, 2.9f },new float[] {6.7f, 3.1f },new float[] {5.6f, 3.0f },
                            new float[] {5.8f, 2.7f },new float[] {6.2f, 2.2f },new float[] {5.6f, 2.5f },
                            new float[] {5.9f, 3.0f},new float[] {6.1f, 2.8f},new float[] {6.3f, 2.5f},
                            new float[] {6.1f, 2.8f},new float[] {6.4f, 2.9f},new float[] {6.6f, 3.0f },
                            new float[] {6.8f, 2.8f},new float[] {6.7f, 3.0f },new float[] {6.0f , 2.9f},
                            new float[] {5.7f, 2.6f},new float[] {5.5f, 2.4f},new float[] {5.5f, 2.4f},
                            new float[] {5.8f, 2.7f},new float[] {6.0f , 2.7f},new float[] {5.4f, 3.0f},
                            new float[] {6.0f , 3.4f},new float[] {6.7f, 3.1f},new float[] {6.3f, 2.3f},
                            new float[] {5.6f, 3.0f },new float[] {5.5f, 2.5f},new float[] {5.5f, 2.6f},
                            new float[] {6.1f, 3.0f },new float[] {5.8f, 2.6f},new float[] {5.0f, 2.3f},
                            new float[] {5.6f, 2.7f},new float[] {5.7f, 3.0f },new float[] {5.7f, 2.9f},
                            new float[] {6.2f, 2.9f},new float[] {5.1f, 2.5f},new float[] {5.7f, 2.8f},
                            new float[] {6.3f, 3.3f},new float[] {5.8f, 2.7f},new float[] {7.1f, 3.0f },
                            new float[] {6.3f, 2.9f},new float[] {6.5f, 3.0f },new float[] {7.6f, 3.0f },
                            new float[] {4.9f, 2.5f},new float[] {7.3f, 2.9f},new float[] {6.7f, 2.5f},
                            new float[] {7.2f, 3.6f},new float[] {6.5f, 3.2f},new float[] {6.4f, 2.7f},
                            new float[] {6.8f, 3.00f },new float[] {5.7f, 2.5f},new float[] {5.8f, 2.8f},
                            new float[] {6.4f, 3.2f},new float[] {6.5f, 3.0f },new float[] {7.7f, 3.8f},
                            new float[] {7.7f, 2.6f},new float[] {6.0f , 2.2f},new float[] {6.9f, 3.2f},
                            new float[] {5.6f, 2.8f},new float[] {7.7f, 2.8f},new float[] {6.3f, 2.7f},
                            new float[] {6.7f, 3.3f},new float[] {7.2f, 3.2f},new float[] {6.2f, 2.8f},
                            new float[] {6.1f, 3.0f },new float[] {6.4f, 2.8f},new float[] {7.2f, 3.0f },
                            new float[] {7.4f, 2.8f},new float[] {7.9f, 3.8f},new float[] {6.4f, 2.8f},
                            new float[] {6.3f, 2.8f},new float[] {6.1f, 2.6f},new float[] {7.7f, 3.0f },
                            new float[] {6.3f, 3.4f},new float[] {6.4f, 3.1f},new float[] {6.0f, 3.0f},
                            new float[] {6.9f, 3.1f},new float[] {6.7f, 3.1f},new float[] {6.9f, 3.1f},
                            new float[] {5.8f, 2.7f},new float[] {6.8f, 3.2f},new float[] {6.7f, 3.3f},
                            new float[] {6.7f, 3.0f },new float[] {6.3f, 2.5f},new float[] {6.5f, 3.0f },
                            new float[] {6.2f, 3.4f},new float[] {5.9f, 3.0f }, new float[] {5.8f, 3.0f }});
            
            var y = np.array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
            fit(X, y);
            // Create a regular grid and classify each point 
            float x_min = X.amin(0).Data<float>(0) - 0.5f;
            float y_min = X.amin(0).Data<float>(1) - 0.5f;
            float x_max = X.amax(0).Data<float>(0) + 0.5f;
            float y_max = X.amax(0).Data<float>(1) + 0.5f;

            var (xx, yy) = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30));
            var s = tf.Session();
            if (xx.dtype ==  typeof(float))
            {
                var samples = np.hstack<float>(xx.ravel().reshape(xx.size,1), yy.ravel().reshape(yy.size,1));
                var Z = s.run(predict(samples));
            }

            return true;
        }

        public void fit(NDArray X, NDArray y)
        {
            var unique_y = y.unique<int>();

            var dic = new Dictionary<int, List<List<float>>>();
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
                var curClass = y[i];
                var l = dic[curClass];
                var pair = new List<float>();
                pair.Add(X[i,0]);
                pair.Add(X[i, 1]);
                l.Add(pair);
                if (l.Count > maxCount)
                {
                    maxCount = l.Count;
                }
                dic[curClass] = l;
            }
            float[,,] points = new float[dic.Count, maxCount, X.shape[1]];
            foreach (KeyValuePair<int, List<List<float>>> kv in dic)
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
            var points_by_class = np.array(points);
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
            var t1= ops.convert_to_tensor(X, TF_DataType.TF_FLOAT);
            var t2 = ops.convert_to_tensor(new int[] { 1, nb_classes });
            Tensor tile = tf.tile(t1, t2);
            var t3 = ops.convert_to_tensor(new int[] { -1, nb_classes, nb_features });
            Tensor r = tf.reshape(tile, t3);
            var cond_probs = tf.reduce_sum(dist.log_prob(r), 2);
            // uniform priors
            float[] tem = new float[nb_classes];
            for (int i = 0; i < tem.Length; i++)
            {
                tem[i] = 1.0f / nb_classes;
            }
            var priors = np.log(np.array<float>(tem));

            // posterior log probability, log P(c) + log P(x|c)
            var joint_likelihood = tf.add(ops.convert_to_tensor(priors, TF_DataType.TF_FLOAT), cond_probs);
            // normalize to get (log)-probabilities

            var norm_factor = tf.reduce_logsumexp(joint_likelihood, new int[] { 1 }, keepdims: true);
            var log_prob = joint_likelihood - norm_factor;
            // exp to get the actual probabilities
            return tf.exp(log_prob);
        }

        public void PrepareData()
        {
            
        }
    }
}
