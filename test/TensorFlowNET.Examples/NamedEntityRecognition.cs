using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://github.com/guillaumegenthial/tf_ner
    /// </summary>
    public class NamedEntityRecognition : IExample
    {
        public int Priority => 100;
        public bool Enabled { get; set; } = false;
        public string Name => "NER";
        public bool ImportGraph { get; set; } = false;


        public bool Run()
        {
            throw new NotImplementedException();
        }

        public void PrepareData()
        {
            throw new NotImplementedException();
        }
    }
}
