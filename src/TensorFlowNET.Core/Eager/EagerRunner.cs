using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Gradients;

namespace Tensorflow.Eager
{
    public class EagerRunner : IEagerRunner
    {
        public Tensor[] TFE_Execute(Context ctx, string device_name, string op_name, Tensor[] inputs, object[] attrs, int num_outputs)
        {
            throw new NotImplementedException();
        }

        public Tensor[] TFE_FastPathExecute(Context ctx, string device_name, string opName, string name, Action callbacks, params object[] args)
        {
            throw new NotImplementedException();
        }

        public Tensor[] TFE_TapeGradient(ITape tape, Tensor[] target, Tensor[] sources, Tensor[] output_gradients)
        {
            throw new NotImplementedException();
        }
    }
}
