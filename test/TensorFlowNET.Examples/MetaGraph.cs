using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    public class MetaGraph : IExample
    {
        public int Priority => 100;
        public bool Enabled { get; set; } = false;
        public string Name => "Meta Graph";
        public bool ImportGraph { get; set; } = true;


        public bool Run()
        {
            ImportMetaGraph("my-save-dir/");
            return false;
        }

        private void ImportMetaGraph(string dir)
        {
            with(tf.Session(), sess =>
            {
                var new_saver = tf.train.import_meta_graph(dir + "my-model-10000.meta");
                new_saver.restore(sess, dir + "my-model-10000");
                var labels = tf.constant(0, dtype: tf.int32, shape: new int[] { 100 }, name: "labels");
                var batch_size = tf.size(labels);
                var logits = (tf.get_collection("logits") as List<ITensorOrOperation>)[0] as Tensor;
                var loss = tf.losses.sparse_softmax_cross_entropy(labels: labels,
                                                logits: logits);
            });
        }

        public void PrepareData()
        {
        }
    }
}
