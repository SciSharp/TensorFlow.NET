using System.Collections.Generic;
using System.Linq;
using System;
using static Tensorflow.OpDef.Types;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public partial class wrap_tfe_src
    {
        public static IntPtr[] TFE_Execute(Context ctx,
            string device_name,
            string op_name,
            Tensor[] inputs,
            object[] attrs,
            int num_outputs,
            Status status)
             => TFE_ExecuteCancelable(ctx, device_name, op_name, inputs, attrs, num_outputs, status);

        public static IntPtr[] TFE_ExecuteCancelable(Context ctx,
            string device_name,
            string op_name,
            Tensor[] inputs,
            object[] attrs, 
            int num_outputs,
            Status status)
        {
            var op = GetOp(ctx, op_name, status);
            status.Check(true);
            c_api.TFE_OpSetDevice(op, device_name, status);
            if(status.ok())
            {
                for (int i = 0; i < inputs.Length; ++i)
                {
                    TFE_TensorHandle tensor_handle;
                    switch (inputs[i])
                    {
                        case EagerTensor et:
                            tensor_handle = (TFE_TensorHandle)et;
                            break;
                        default:
                            tensor_handle = c_api.TFE_NewTensorHandle(inputs[i], status);
                            break;
                    }
                    c_api.TFE_OpAddInput(op, tensor_handle, status);
                }
            }
            if (status.ok())
                SetOpAttrs(ctx, op, attrs, status);

            var outputs = new IntPtr[num_outputs];
            if (status.ok())
            {
                c_api.TFE_Execute(op, outputs, ref num_outputs, status);
                status.Check(true);
            }
            return outputs;
        }
    }
}
