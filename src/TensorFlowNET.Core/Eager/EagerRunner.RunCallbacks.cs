namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        bool RunCallbacks(FastPathOpExecInfo op_exec_info,
            int num_inferred_attrs,
            Tensor[] inputs,
            object[] attrs,
            Tensor[] flattened_result)
        {
            if (op_exec_info.run_gradient_callback)
            {
                if (!RecordGradient(op_exec_info.op_name, inputs, attrs,
                                    flattened_result))
                {
                    return false;
                }
            }

            if (op_exec_info.run_post_exec_callbacks)
            {

            }

            return true;
        }
    }
}
