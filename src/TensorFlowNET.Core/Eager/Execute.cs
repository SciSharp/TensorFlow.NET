using System.Collections.Generic;
using System;
using System.Linq;

namespace Tensorflow.Eager
{
    public class Execute
    {
        /// <summary>
        /// Execute a TensorFlow operation.
        /// </summary>
        /// <param name="op_name">
        /// Name of the TensorFlow operation (see REGISTER_OP in C++ code) to 
        /// execute.
        /// </param>
        /// <param name="num_outputs">
        /// The number of outputs of the operation to fetch.
        /// </param>
        /// <param name="inputs">
        /// A list of inputs to the operation. Each entry should be a Tensor, or
        /// a value which can be passed to the Tensor constructor to create one.
        /// </param>
        /// <param name="attrs">
        /// A tuple with alternating string attr names and attr values for this
        /// operation.
        /// </param>
        /// <param name="ctx">The value of context.context().</param>
        /// <param name="name">Customized name for the operation.</param>
        /// <returns>List of output Tensor objects. The list is empty if there are no outputs</returns>
        public Tensor execute(Context ctx, string op_name, Tensor[] inputs, object[] attrs, string name = null)
        {
            ctx.ensure_initialized();
            using (var status = new Status())
            {
                var retVals = wrap_tfe_src.TFE_Py_Execute(ctx, ctx.device_name, op_name, inputs, attrs, 1, status);

                var t = c_api.TFE_TensorHandleResolve(retVals[0], status);
                status.Check(true);

                return new EagerTensor(t);
            }
        }

        public (TF_DataType, Tensor) args_to_matching_eager(Tensor[] l, Context ctx, TF_DataType default_dtype = TF_DataType.DtInvalid)
        {
            var dtype = default_dtype;
            if(dtype == TF_DataType.DtInvalid)
            {
                var tensor = ops.convert_to_tensor(l, dtype, preferred_dtype: default_dtype, ctx: ctx);

                if (dtype == TF_DataType.DtInvalid)
                    dtype = tensor.dtype;

                return (dtype, tensor);
            }
            else
            {
                return (dtype, l[0]);
            }
        }

        public void record_gradient(string op_name, InputList inputs, Dictionary<string, object> attrs, Tensor[] results, string name = null)
        {
            wrap_tfe_src.RecordGradient(op_name, inputs._inputs, attrs, results, name);
        }
    }
}
