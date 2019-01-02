using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    /// <summary>
    /// tensorflow\c\c_test_util.cc
    /// </summary>
    public class CSession
    {
        private IntPtr session_;

        private List<IntPtr> inputs_ = new List<IntPtr>();
        private List<IntPtr> input_values_ = new List<IntPtr>();
        private List<IntPtr> outputs_ = new List<IntPtr>();
        private List<IntPtr> output_values_ = new List<IntPtr>();

        private List<IntPtr> targets_ = new List<IntPtr>();

        public CSession(Graph graph, Status s, bool user_XLA = false)
        {
            var opts = new SessionOptions();
            session_ = new Session(graph, opts, s);
        }

        public void SetInputs(Dictionary<IntPtr, IntPtr> inputs)
        {
            DeleteInputValues();
            inputs_.Clear();
            foreach (var input in inputs)
            {
                var handle = Marshal.AllocHGlobal(Marshal.SizeOf<TF_Output>());
                Marshal.StructureToPtr(new TF_Output(input.Key, 0), handle, false);
                inputs_.Add(handle);

                input_values_.Add(input.Value);
            }
        }

        private void DeleteInputValues()
        {
            for (var i = 0; i < input_values_.Count; ++i)
            {
                //input_values_[i].Dispose();
            }
            input_values_.Clear();
        }

        public void SetOutputs(List<IntPtr> outputs)
        {
            ResetOutputValues();
            outputs_.Clear();
            foreach (var output in outputs)
            {
                var handle = Marshal.AllocHGlobal(Marshal.SizeOf<TF_Output>());
                Marshal.StructureToPtr(new TF_Output(output, 0), handle, true);
                outputs_.Add(handle);
                handle = Marshal.AllocHGlobal(Marshal.SizeOf<IntPtr>());
                output_values_.Add(IntPtr.Zero);
            }
        }

        private void ResetOutputValues()
        {
            for (var i = 0; i < output_values_.Count; ++i)
            {
                //if (output_values_[i] != IntPtr.Zero)
                    //output_values_[i].Dispose();
            }
            output_values_.Clear();
        }

        public unsafe void Run(Status s)
        {
            IntPtr inputs_ptr = inputs_.Count == 0 ? IntPtr.Zero : inputs_[0];
            IntPtr input_values_ptr = inputs_.Count == 0 ? IntPtr.Zero : input_values_[0];
            IntPtr outputs_ptr = outputs_.Count == 0 ? IntPtr.Zero : outputs_[0];
            IntPtr[] output_values_ptr = output_values_.ToArray();// output_values_.Count == 0 ? IntPtr.Zero : output_values_[0];
            IntPtr targets_ptr = IntPtr.Zero;

            c_api.TF_SessionRun(session_, null, inputs_ptr, input_values_ptr, 0,
                outputs_ptr, output_values_ptr, outputs_.Count, 
                targets_ptr, targets_.Count,
                IntPtr.Zero, s);

            s.Check();

            output_values_[0] = output_values_ptr[0];
        }

        public IntPtr output_tensor(int i)
        {
            return output_values_[i];
        }
    }
}
