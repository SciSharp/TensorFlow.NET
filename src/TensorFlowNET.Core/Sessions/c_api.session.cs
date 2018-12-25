using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public static partial class c_api
    {
        /// <summary>
        /// Destroy an options object.
        /// </summary>
        /// <param name="opts"></param>
        [DllImport(TensorFlowLibName)]
        public static unsafe extern void TF_DeleteSessionOptions(IntPtr opts);

        /// <summary>
        /// Return a new execution session with the associated graph, or NULL on
        /// error. Does not take ownership of any input parameters.
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="opts"></param>
        /// <param name="status"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewSession(IntPtr graph, IntPtr opts, IntPtr status);

        /// <summary>
        /// Return a new options object.
        /// </summary>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewSessionOptions();

        /// <summary>
        /// Run the graph associated with the session starting with the supplied inputs
        /// (inputs[0,ninputs-1] with corresponding values in input_values[0,ninputs-1]).
        /// </summary>
        /// <param name="session"></param>
        /// <param name="run_options"></param>
        /// <param name="inputs"></param>
        /// <param name="input_values"></param>
        /// <param name="ninputs"></param>
        /// <param name="outputs"></param>
        /// <param name="output_values"></param>
        /// <param name="noutputs"></param>
        /// <param name="target_opers"></param>
        /// <param name="ntargets"></param>
        /// <param name="run_metadata"></param>
        /// <param name="status"></param>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SessionRun(IntPtr session, IntPtr run_options,
                   TF_Output[] inputs, IntPtr[] input_values, int ninputs,
                   TF_Output[] outputs, IntPtr[] output_values, int noutputs,
                   IntPtr[] target_opers, int ntargets,
                   IntPtr run_metadata,
                   IntPtr status);
    }
}
