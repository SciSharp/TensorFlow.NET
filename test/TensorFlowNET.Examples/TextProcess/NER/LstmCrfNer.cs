using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Estimator;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples.Text.NER
{
    /// <summary>
    /// A NER model using Tensorflow (LSTM + CRF + chars embeddings).
    /// State-of-the-art performance (F1 score between 90 and 91).
    /// 
    /// https://github.com/guillaumegenthial/sequence_tagging
    /// </summary>
    public class LstmCrfNer : IExample
    {
        public int Priority => 14;

        public bool Enabled { get; set; } = true;
        public bool ImportGraph { get; set; } = true;

        public string Name => "LSTM + CRF NER";

        HyperParams hp;
        
        Dictionary<string, int> vocab_tags = new Dictionary<string, int>();
        int nwords, nchars, ntags;
        CoNLLDataset dev, train;

        public bool Run()
        {
            PrepareData();
            var graph = tf.Graph().as_default();

            tf.train.import_meta_graph("graph/lstm_crf_ner.meta");

            var init = tf.global_variables_initializer();

            with(tf.Session(), sess =>
            {
                sess.run(init);

                foreach (var epoch in range(hp.epochs))
                {
                    print($"Epoch {epoch + 1} out of {hp.epochs}");
                }

            });

            return true;
        }

        public void PrepareData()
        {
            hp = new HyperParams("LstmCrfNer")
            {
                epochs = 15,
                dropout = 0.5f,
                batch_size = 20,
                lr_method = "adam",
                lr = 0.001f,
                lr_decay = 0.9f,
                clip = false,
                epoch_no_imprv = 3,
                hidden_size_char = 100,
                hidden_size_lstm = 300
            };
            hp.filepath_dev = hp.filepath_test = hp.filepath_train = Path.Combine(hp.data_root_dir, "test.txt");

            // Loads vocabulary, processing functions and embeddings
            hp.filepath_words = Path.Combine(hp.data_root_dir, "words.txt");
            hp.filepath_tags = Path.Combine(hp.data_root_dir, "tags.txt");
            hp.filepath_chars = Path.Combine(hp.data_root_dir, "chars.txt");

            // 1. vocabulary
            /*vocab_tags = load_vocab(hp.filepath_tags);
            

            nwords = vocab_words.Count;
            nchars = vocab_chars.Count;
            ntags = vocab_tags.Count;*/

            // 2. get processing functions that map str -> id
            dev = new CoNLLDataset(hp.filepath_dev, hp);
            train = new CoNLLDataset(hp.filepath_train, hp);
        }
    }
}
