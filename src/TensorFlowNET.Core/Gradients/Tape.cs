using System;
using System.Collections.Generic;
using Tensorflow.Util;
using Microsoft.Extensions.Logging;
using static Tensorflow.Binding;
using static Tensorflow.tensorflow;

namespace Tensorflow.Gradients
{
    public partial class Tape : ITape
    {
        int nesting_id;
        static int tape_nesting_id_counter = 0;
        bool persistent_;
        bool watch_accessed_variables;
        TensorTape tensor_tape_;
        OpTape<BackwardFunction, TapeTensor> op_tape_;

        /// <summary>
        /// A deque-backed stack, whose element references are not invalidated by
        /// pushes and pops at the back.
        /// </summary>
        Stack<AccumulatorCallState> call_state_;

        public Tape(bool persistent, bool watch_accessed_variables)
        {
            this.persistent_ = persistent;
            this.watch_accessed_variables = watch_accessed_variables;

            tensor_tape_ = new TensorTape();
            op_tape_ = new OpTape<BackwardFunction, TapeTensor>();
            tensor_usage_ = new UnorderedMap<long, long>();

            nesting_id = ++tape_nesting_id_counter;
            tf.GetTapeSet().Add(this);
        }

        /// <summary>
        /// Marks this tensor to be watched by the given tape.
        /// </summary>
        /// <param name="x"></param>
        public void Watch(long tensor_id)
        {
            if (!CouldBackprop())
                return;

            tf.Logger.LogDebug($"Watch tensor_id={tensor_id}");
            tensor_tape_.emplace(tensor_id, -1);
        }

        public bool ShouldRecord(long[] tensor_ids, TF_DataType[] dtypes)
        {
            for (int i = 0; i < tensor_ids.Length; ++i)
            {
                if (tensor_tape_.find(tensor_ids[i]))
                {
                    if (IsDtypeTrainable(dtypes[i]))
                    {
                        tf.Logger.LogDebug($"tape.h->ShouldRecord: should_record = true, tensor_tape_.size()={tensor_tape_.Count}, tensor_ids[{i}]={tensor_ids[i]}");
                        return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Pops the given tape in the stack.
        /// </summary>
        /// <param name="tape"></param>
        public void PopTape(ITape tape)
        {
            tf.GetTapeSet().Remove(tape);
        }

        public void VariableAccessed(ResourceVariable variable)
        {
            Watch(variable.Handle.Id);
        }

        public ResourceVariable[] WatchedVariables()
        {
            return null;
        }

        public bool IsDtypeTrainable(TF_DataType dtype)
        {
            switch (dtype)
            {
                case TF_DataType.TF_HALF:
                case TF_DataType.TF_BFLOAT16:
                case TF_DataType.TF_FLOAT:
                case TF_DataType.TF_DOUBLE:
                case TF_DataType.TF_COMPLEX64:
                case TF_DataType.TF_COMPLEX128:
                case TF_DataType.TF_RESOURCE:
                case TF_DataType.TF_VARIANT:
                    return true;
                default:
                    return false;
            }
        }

        bool CouldForwardprop()
            => HasAccumulator();

        bool CouldBackprop()
            => HasGradientTape();

        bool HasAccumulator()
            //return !GetAccumulatorSet()->empty();
            => false;

        bool HasGradientTape()
            => tf.GetTapeSet().Count > 0;
    }
}
