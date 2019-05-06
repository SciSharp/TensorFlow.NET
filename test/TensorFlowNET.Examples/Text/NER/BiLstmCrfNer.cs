using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Bidirectional LSTM-CRF Models for Sequence Tagging 
    /// https://github.com/guillaumegenthial/tf_ner/tree/master/models/lstm_crf
    /// </summary>
    public class BiLstmCrfNer : Python, IExample
    {
        public int Priority => 13;

        public bool Enabled { get; set; } = true;
        public bool ImportGraph { get; set; } = false;

        public string Name => "bi-LSTM + CRF NER";
        HyperParams @params = new HyperParams();

        public bool Run()
        {
            PrepareData();
            return true;
        }

        public void PrepareData()
        {
            if (!Directory.Exists(HyperParams.DATADIR))
                Directory.CreateDirectory(HyperParams.DATADIR);

            if (!Directory.Exists(@params.RESULTDIR))
                Directory.CreateDirectory(@params.RESULTDIR);

            if (!Directory.Exists(@params.MODELDIR))
                Directory.CreateDirectory(@params.MODELDIR);

            if (!Directory.Exists(@params.EVALDIR))
                Directory.CreateDirectory(@params.EVALDIR);
        }

        private class HyperParams
        {
            public const string DATADIR = "BiLstmCrfNer";
            public string RESULTDIR = Path.Combine(DATADIR, "results");
            public string MODELDIR;
            public string EVALDIR;

            public int dim = 300;
            public float dropout = 0.5f;
            public int num_oov_buckets = 1;
            public int epochs = 25;
            public int batch_size = 20;
            public int buffer = 15000;
            public int lstm_size = 100;
            public string words = Path.Combine(DATADIR, "vocab.words.txt");
            public string chars = Path.Combine(DATADIR, "vocab.chars.txt");
            public string tags = Path.Combine(DATADIR, "vocab.tags.txt");
            public string glove = Path.Combine(DATADIR, "glove.npz");

            public HyperParams()
            {
                MODELDIR = Path.Combine(RESULTDIR, "model");
                EVALDIR = Path.Combine(MODELDIR, "eval");
            }
        }
    }
}
