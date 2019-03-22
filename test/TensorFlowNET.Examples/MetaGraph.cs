using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    public class MetaGraph : Python, IExample
    {
        public void Run()
        {
            ImportMetaGraph("my-save-dir/");
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
