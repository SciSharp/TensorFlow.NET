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

    // from ops.py
    public partial class Operation
    {
        public TF_Output Input(int index) => c_api.TF_OperationInput(new TF_Input(_handle, index));
        public TF_DataType InputType(int index) => c_api.TF_OperationInputType(new TF_Input(_handle, index));
        public int InputListLength(string name) => c_api.TF_OperationInputListLength(_handle, name, status);
        public int NumInputs => c_api.TF_OperationNumInputs(_handle);
        private TF_DataType[] _input_types => _inputs._inputs.Select(x => x.dtype).ToArray();

        private InputList _inputs;
        public InputList inputs
        {
            get
            {
                if (_inputs == null)
                {
                    var retval = new Tensor[NumInputs];

                    for (int i = 0; i < NumInputs; i++)
                    {
                        var tf_output = Input(i);
                        var op = new Operation(tf_output.oper);
                        retval[i] = op.outputs[tf_output.index];
                    }

                    _inputs = new InputList(retval);
                }

                return _inputs;
            }
        }

        public int NumControlInputs => c_api.TF_OperationNumControlInputs(_handle);

        /// <summary>
        /// The `Operation` objects on which this op has a control dependency.
        /// 
        /// Before this op is executed, TensorFlow will ensure that the
        /// operations in `self.control_inputs` have finished executing.This
        /// mechanism can be used to run ops sequentially for performance
        /// reasons, or to ensure that the side effects of an op are observed
        /// in the correct order.
        /// </summary>
        public Operation[] control_inputs
        {
            get
            {
                return GetControlInputs();
            }
        }

        public unsafe Operation[] GetControlInputs()
        {
            var control_inputs = new Operation[NumControlInputs];

            if (NumControlInputs > 0)
            {
                IntPtr control_input_handle = Marshal.AllocHGlobal(Marshal.SizeOf<IntPtr>() * NumControlInputs);
                c_api.TF_OperationGetControlInputs(_handle, control_input_handle, NumControlInputs);
                for (int i = 0; i < NumControlInputs; i++)
                {
                    var handle = control_input_handle + Marshal.SizeOf<IntPtr>() * i;
                    control_inputs[i] = new Operation(*(IntPtr*)handle);
                }
            }

            return control_inputs;
        }
    }
}
