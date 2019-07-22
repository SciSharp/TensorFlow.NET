using System;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// The CRF module implements a linear-chain CRF layer for learning to predict tag sequences.
    /// This variant of the CRF is factored into unary potentials for every element 
    /// in the sequence and binary potentials for every transition between output tags.
    /// 
    /// tensorflow\contrib\crf\python\ops\crf.py
    /// </summary>
    public class CRF : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = false;

        public string Name => "CRF";

        public bool Run()
        {
            return true;
        }

        public void PrepareData()
        {

        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public void Train(Session sess)
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
