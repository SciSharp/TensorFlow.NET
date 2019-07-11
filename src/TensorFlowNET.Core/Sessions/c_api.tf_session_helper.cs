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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        public static string[] TF_OperationOutputConsumers_wrapper(TF_Output oper_out)
        {
            int num_consumers = TF_OperationOutputNumConsumers(oper_out);
            int size = Marshal.SizeOf<TF_Input>();
            var handle = Marshal.AllocHGlobal(size * num_consumers);
            int num = TF_OperationOutputConsumers(oper_out, handle, num_consumers);
            var consumers = new string[num_consumers];
            for (int i = 0; i < num; i++)
            {
                TF_Input input = Marshal.PtrToStructure<TF_Input>(handle + i * size);
                consumers[i] = Marshal.PtrToStringAnsi(TF_OperationName(input.oper));
            }

            return consumers;
        }
    }
}
