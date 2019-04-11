using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://github.com/guillaumegenthial/tf_ner
    /// </summary>
    public class NamedEntityRecognition : Python, IExample
    {
        public int Priority => 100;
        public bool Enabled { get; set; } = false;
        public string Name => "NER";

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
