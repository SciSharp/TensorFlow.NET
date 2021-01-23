using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Util;

namespace Tensorflow.Native.UnitTest
{
    /// <summary>
    /// tensorflow\c\c_test_util.cc
    /// TEST(CAPI, Session)
    /// </summary>
    public class CSession
    {
        private IntPtr session_;

        private List<TF_Output> inputs_ = new List<TF_Output>();
        private List<Tensor> input_values_ = new List<Tensor>();
        private List<TF_Output> outputs_ = new List<TF_Output>();
        private List<Tensor> output_values_ = new List<Tensor>();

        private List<IntPtr> targets_ = new List<IntPtr>();

        public CSession(Graph graph, Status s, bool user_XLA = false)
        {
            lock (Locks.ProcessWide)
            {
                var config = new ConfigProto { InterOpParallelismThreads = 4 };
                session_ = new Session(graph, config, s);
            }
        }

        public void SetInputs(Dictionary<Operation, Tensor> inputs)
        {
            DeleteInputValues();
            inputs_.Clear();
            foreach (var input in inputs)
            {
                inputs_.Add(new TF_Output(input.Key, 0));
                input_values_.Add(input.Value);
            }
        }

        public void SetInputs(KeyValuePair<Operation, Tensor>[] inputs)
        {
            DeleteInputValues();
            inputs_.Clear();
            foreach (var input in inputs)
            {
                inputs_.Add(new TF_Output(input.Key, 0));
                input_values_.Add(input.Value);
            }
        }

        private void DeleteInputValues()
        {
            //clearing is enough as they will be disposed by the GC unless they are referenced else-where.
            input_values_.Clear();
        }

        public void SetOutputs(TF_Output[] outputs)
        {
            ResetOutputValues();
            outputs_.Clear();
            foreach (var output in outputs)
            {
                outputs_.Add(output);
                output_values_.Add(IntPtr.Zero);
            }
        }

        private void ResetOutputValues()
        {
            //clearing is enough as they will be disposed by the GC unless they are referenced else-where.
            output_values_.Clear();
        }

        public unsafe void Run(Status s)
        {
            var inputs_ptr = inputs_.ToArray();
            var input_values_ptr = input_values_.Select(x => (IntPtr)x).ToArray();
            var outputs_ptr = outputs_.ToArray();
            var output_values_ptr = output_values_.Select(x => IntPtr.Zero).ToArray();
            IntPtr[] targets_ptr = new IntPtr[0];

            c_api.TF_SessionRun(session_, null, inputs_ptr, input_values_ptr, inputs_ptr.Length,
                outputs_ptr, output_values_ptr, outputs_.Count,
                targets_ptr, targets_.Count,
                IntPtr.Zero, s.Handle);

            s.Check();

            for (var i = 0; i < outputs_.Count; i++)
                output_values_[i] = output_values_ptr[i];
        }

        public IntPtr output_tensor(int i)
        {
            return output_values_[i];
        }

        public void CloseAndDelete(Status s)
        {
            DeleteInputValues();
            ResetOutputValues();
        }
    }
}