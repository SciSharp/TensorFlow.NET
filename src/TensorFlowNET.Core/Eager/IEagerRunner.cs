using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Contexts;
using Tensorflow.Gradients;

namespace Tensorflow.Eager
{
    public interface IEagerRunner
    {
        Tensor[] Execute(Context ctx, string op_name,
            int num_outputs,
            Tensor[] inputs, object[] attrs,
            string name = null);

        (TF_DataType, Tensor[]) ArgsToMatchingEager(Context ctx, 
            TF_DataType default_dtype = TF_DataType.DtInvalid, 
            object[] args = null);

        Tensor[] TFE_FastPathExecute(Context ctx,
            string device_name,
            string opName,
            string name,
            Action callbacks,
            params object[] args);

        Tensor[] TFE_Execute(Context ctx,
            string device_name,
            string op_name,
            Tensor[] inputs,
            object[] attrs,
            int num_outputs);

        Tensor[] TFE_TapeGradient(ITape tape,
            Tensor[] target,
            Tensor[] sources,
            Tensor[] output_gradients);

        bool RecordGradient(string op_name,
            Tensor[] inputs,
            object[] attrs,
            Tensor[] results);
    }
}
