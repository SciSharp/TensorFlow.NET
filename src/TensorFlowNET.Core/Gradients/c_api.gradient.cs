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
using System.Runtime.InteropServices;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Adds operations to compute the partial derivatives of sum of `y`s w.r.t `x`s,
        /// i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...
        /// This is a variant of TF_AddGradients that allows to caller to pass a custom
        /// name prefix to the operations added to a graph to compute the gradients.
        /// </summary>
        /// <param name="g">TF_Graph*</param>
        /// <param name="prefix">const char*</param>
        /// <param name="y">TF_Output*</param>
        /// <param name="ny">int</param>
        /// <param name="x">TF_Output*</param>
        /// <param name="nx">int</param>
        /// <param name="dx">TF_Output*</param>
        /// <param name="status">TF_Status*</param>
        /// <param name="dy">TF_Output*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_AddGradientsWithPrefix(SafeGraphHandle g, string prefix, TF_Output[] y, int ny,
            TF_Output[] x, int nx, TF_Output[] dx, SafeStatusHandle status, IntPtr[] dy);
    }
}
