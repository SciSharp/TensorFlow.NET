﻿/*****************************************************************************
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

namespace Tensorflow
{
    public partial class Operation
    {
        public int NumOutputs => c_api.TF_OperationNumOutputs(_handle);
        public TF_DataType OutputType(int index) => c_api.TF_OperationOutputType(new TF_Output(_handle, index));
        public int OutputListLength(string name) => c_api.TF_OperationOutputListLength(_handle, name, status);

        private Tensor[] _outputs;
        public Tensor[] outputs => _outputs;

        public Tensor output => _outputs.FirstOrDefault();

        public int NumControlOutputs => c_api.TF_OperationNumControlOutputs(_handle);

        public int OutputNumConsumers(int index) => c_api.TF_OperationOutputNumConsumers(new TF_Output(_handle, index));

        public unsafe TF_Input[] OutputConsumers(int index, int max_consumers)
        {
            int size = Marshal.SizeOf<TF_Input>();
            var handle = Marshal.AllocHGlobal(size);
            int num = c_api.TF_OperationOutputConsumers(new TF_Output(_handle, index), handle, max_consumers);
            var consumers = new TF_Input[num];
            for (int i = 0; i < num; i++)
            {
                consumers[i] = Marshal.PtrToStructure<TF_Input>(handle + i * size);
            }

            return consumers;
        }

        public unsafe Operation[] GetControlOutputs()
        {
            var control_outputs = new Operation[NumControlOutputs];

            if (NumControlOutputs > 0)
            {
                IntPtr control_output_handle = Marshal.AllocHGlobal(Marshal.SizeOf<IntPtr>() * NumControlOutputs);
                c_api.TF_OperationGetControlOutputs(_handle, control_output_handle, NumControlOutputs);
                for (int i = 0; i < NumControlOutputs; i++)
                {
                    var handle = control_output_handle + Marshal.SizeOf<IntPtr>() * i;
                    control_outputs[i] = new Operation(*(IntPtr*)handle);
                }
            }

            return control_outputs;
        }
    }
}
