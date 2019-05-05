using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    public class BiLstmCrfNer : Python, IExample
    {
        public int Priority => 13;

        public bool Enabled { get; set; } = true;
        public bool ImportGraph { get; set; } = false;

        public string Name => "bi-LSTM + CRF NER";

        public void PrepareData()
        {
            
        }

        public bool Run()
        {
            return true;
        }
    }
}
