using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Return a new options object.
        /// </summary>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewContextOptions();

        /// <summary>
        /// Destroy an options object.
        /// </summary>
        /// <param name="options"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteContextOptions(IntPtr options);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewContext(IntPtr opts, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteContext(IntPtr ctx);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewOp(IntPtr ctx, string op_or_function_name, IntPtr status);
    }
}
