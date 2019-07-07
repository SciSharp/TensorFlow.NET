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
        public bool Enabled { get; set; } = false;
        public string Name => "NER";
        public bool IsImportingGraph { get; set; } = false;


        public void Train(Session sess)
        {
            throw new NotImplementedException();
        }

        public void PrepareData()
        {
            throw new NotImplementedException();
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public bool Run()
        {
            throw new NotImplementedException();
        }

        public void Predict(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }
    }
}
