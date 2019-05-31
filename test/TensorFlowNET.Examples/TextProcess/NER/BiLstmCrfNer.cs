using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using Tensorflow.Estimator;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Bidirectional LSTM-CRF Models for Sequence Tagging 
    /// https://github.com/guillaumegenthial/tf_ner/tree/master/models/lstm_crf
    /// </summary>
    public class BiLstmCrfNer : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = false;

        public string Name => "bi-LSTM + CRF NER";

        public bool Run()
        {
            PrepareData();
            return false;
        }

        public void PrepareData()
        {
            var hp = new HyperParams("BiLstmCrfNer");
            hp.filepath_words = Path.Combine(hp.data_root_dir, "vocab.words.txt");
            hp.filepath_chars = Path.Combine(hp.data_root_dir, "vocab.chars.txt");
            hp.filepath_tags = Path.Combine(hp.data_root_dir, "vocab.tags.txt");
            hp.filepath_glove = Path.Combine(hp.data_root_dir, "glove.npz");
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
