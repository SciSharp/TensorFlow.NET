using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Implement Word2Vec algorithm to compute vector representations of words.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/word2vec.py
    /// </summary>
    public class Word2Vec : Python, IExample
    {
        public int Priority => 12;
        public bool Enabled { get; set; } = true;
        public string Name => "Word2Vec";

        // Training Parameters
        float learning_rate = 0.1f;
        int batch_size = 128;
        int num_steps = 3000000;
        int display_step = 10000;
        int eval_step = 200000;

        // Evaluation Parameters
        string[] eval_words = new string[] { "five", "of", "going", "hardware", "american", "britain" };

        public bool Run()
        {
            PrepareData();

            var graph = tf.Graph().as_default();

            tf.train.import_meta_graph("graph/word2vec.meta");

            return false;
        }

        public void PrepareData()
        {
            var url = "";
        }
    }
}
