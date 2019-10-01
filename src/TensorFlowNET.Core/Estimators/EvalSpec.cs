using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    public class EvalSpec
    {
        string _name;

        public EvalSpec(string name, Action input_fn, FinalExporter exporters)
        {
            _name = name;
        }
    }
}
