using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Python;

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
        public int Priority => 13;

        public bool Enabled { get; set; } = true;
        public bool ImportGraph { get; set; } = false;

        public string Name => "CRF";

        public bool Run()
        {
            return true;
        }

        public void PrepareData()
        {

        }
    }
}
