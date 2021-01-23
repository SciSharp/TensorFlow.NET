/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using System.Linq;
using Tensorflow.Contexts;
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
            var status = tf.Status;
            using var op = GetOp(ctx, op_name, status);
            c_api.TFE_OpSetDevice(op, device_name, status.Handle);
            if (status.ok())
            {
                for (int i = 0; i < inputs.Length; ++i)
                {
                    SafeTensorHandleHandle tensor_handle = inputs[i] switch
                    {
                        EagerTensor et => et.EagerTensorHandle,
                        _ => throw new NotImplementedException("")
                    };
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