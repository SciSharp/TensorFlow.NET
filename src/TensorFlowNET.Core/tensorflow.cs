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
using System.Runtime.InteropServices;
using System.Threading;
using Tensorflow.Eager;

namespace Tensorflow
{
    public partial class tensorflow : ITensorFlowObject
    {
        public TF_DataType byte8 = TF_DataType.TF_UINT8;
        public TF_DataType int8 = TF_DataType.TF_INT8;
        public TF_DataType int16 = TF_DataType.TF_INT16;
        public TF_DataType int32 = TF_DataType.TF_INT32;
        public TF_DataType int64 = TF_DataType.TF_INT64;
        public TF_DataType float16 = TF_DataType.TF_HALF;
        public TF_DataType float32 = TF_DataType.TF_FLOAT;
        public TF_DataType float64 = TF_DataType.TF_DOUBLE;
        public TF_DataType @bool = TF_DataType.TF_BOOL;
        public TF_DataType chars = TF_DataType.TF_STRING;
        public TF_DataType @string = TF_DataType.TF_STRING;

        public Context context = new Context(new ContextOptions(), new Status());


        public tensorflow()
        {
            _constructThreadingObjects();
            InitGradientEnvironment();
        }

        private unsafe void InitGradientEnvironment()
        {
            var vspace = c_api.VSpace_Handle((shape, dims, dtype) =>
            {
                var ones = constant_op.constant(1.0f, dtype: dtype) as EagerTensor;
                return ones.EagerTensorHandle;
            }, (gradients, num_grads) =>
            {
                var input_grads = new EagerTensor[num_grads];
                for (int i = 0; i < num_grads; i++)
                    input_grads[i] = new EagerTensor(*((IntPtr*)gradients + i));

                var add_n = gen_math_ops.add_n(input_grads);
                return (add_n as EagerTensor).EagerTensorHandle;
            });

            ops.RegisterFromAssembly();
            c_api.TFE_RegisterGradientFunction((op_name, num_inputs, op_inputs, num_attrs, num_outputs, output_grads) =>
            {
                var input_tensors = new EagerTensor[num_inputs];
                for (int i = 0; i < num_inputs; i++)
                    input_tensors[i] = new EagerTensor(*((IntPtr*)op_inputs + i));

                var output_grad_tensors = new EagerTensor[num_outputs];
                for (int i = 0; i < num_outputs; i++)
                    output_grad_tensors[i] = new EagerTensor(*((IntPtr*)output_grads + i));

                var gradients = ops.gradientFunctions[op_name](new EagerOperation
                {
                    NumInputs = num_inputs,
                    Inputs = input_tensors
                }, output_grad_tensors);

                var ret_tensors = Marshal.AllocHGlobal(sizeof(IntPtr) * num_inputs);
                Marshal.Copy(gradients.Select(x => x == null ? IntPtr.Zero : (x as EagerTensor).EagerTensorHandle).ToArray(), 0, ret_tensors, 2);
                // Marshal.FreeHGlobal(ret_tensors);
                return ret_tensors;
            });
        }

        public ResourceVariable Variable<T>(T data,
            bool trainable = true,
            bool validate_shape = true,
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int[] shape = null)
            => new ResourceVariable(data,
                    trainable: trainable,
                    validate_shape: validate_shape,
                    name: name,
                    dtype: dtype,
                    shape: shape);

        public unsafe Tensor placeholder(TF_DataType dtype, TensorShape shape = null, string name = null)
            => gen_array_ops.placeholder(dtype, shape, name);

        public void enable_eager_execution()
        {
            // contex = new Context();
            context.default_execution_mode = Context.EAGER_MODE;
        }

        public string VERSION => c_api.StringPiece(c_api.TF_Version());

        public Session get_default_session()
            => ops.get_default_session();

        public Session Session()
        {
            return new Session().as_default();
        }

        public Session Session(Graph graph, ConfigProto config = null)
        {
            return new Session(graph, config: config).as_default();
        }

        public Session Session(ConfigProto config)
        {
            return new Session(null, config).as_default();
        }

        public void __init__()
        {
            
        }

        public void __enter__()
        {
            
        }

        public void __exit__()
        {
            
        }

        public void __del__()
        {
            
        }

        public void Dispose()
        {
            
        }
    }
}
