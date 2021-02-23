using System;
using Tensorflow.Contexts;
using Tensorflow.Gradients;
using static Tensorflow.tensorflow;

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

        Tensor[] TFE_FastPathExecute(FastPathOpExecInfo op_exec_info);

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
            Tensor[] results,
            Func<BackwardFunction> getBackwardFunction = null);

        bool MustRecordGradient();

        int TapeSetPossibleGradientTypes(params Tensor[] args);
    }
}
