using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public class Execute
    {
        public void record_gradient(string op_name, Tensor[] inputs, Dictionary<string, object> attrs, Tensor[] results, string name = "")
        {
            if (inputs == null)
                inputs = new Tensor[0];

            pywrap_tfe_src.RecordGradient(op_name, inputs, attrs, results, name);
        }
    }
}
