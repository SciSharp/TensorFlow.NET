using System.Collections.Generic;
using System.Linq;
using System;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public partial class EagerRunner
    {
        public Tensor[] TFE_Execute(Context ctx,
            string device_name,
            string op_name,
            Tensor[] inputs,
            object[] attrs,
            int num_outputs)
             => TFE_ExecuteCancelable(ctx, device_name, op_name, inputs, attrs, num_outputs);

        public Tensor[] TFE_ExecuteCancelable(Context ctx,
            string device_name,
            string op_name,
            Tensor[] inputs,
            object[] attrs,
            int num_outputs)
        {
            var status = tf.status;
            var op = GetOp(ctx, op_name, status);
            status.Check(true);
            c_api.TFE_OpSetDevice(op, device_name, status.Handle);
            if (status.ok())
            {
                for (int i = 0; i < inputs.Length; ++i)
                {
                    SafeTensorHandleHandle tensor_handle;
                    switch (inputs[i])
                    {
                        case EagerTensor et:
                            tensor_handle = et.EagerTensorHandle;
                            break;
                        default:
                            tensor_handle = c_api.TFE_NewTensorHandle(inputs[i], status.Handle);
                            break;
                    }
                    c_api.TFE_OpAddInput(op, tensor_handle, status.Handle);
                    status.Check(true);
                }
            }
            if (status.ok() && attrs != null)
                SetOpAttrs(op, attrs);

            var outputs = new SafeTensorHandleHandle[num_outputs];
            if (status.ok())
            {
                c_api.TFE_Execute(op, outputs, out num_outputs, status.Handle);
                status.Check(true);
            }
            return outputs.Select(x => new EagerTensor(x)).ToArray();
        }
    }
}