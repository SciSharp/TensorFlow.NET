using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Gradients
{
    public partial class Tape : ITape
    {
        int _id;
        // static int tape_nesting_id_counter = 0;
        bool _persistent;
        public bool Persistent => _persistent;
        bool _recording;
        bool _created_eagerly;
        TensorTape tensor_tape_;
        OpTape op_tape_;
        
        /// <summary>
        /// A deque-backed stack, whose element references are not invalidated by
        /// pushes and pops at the back.
        /// </summary>
        // Stack<AccumulatorCallState> call_state_;

        public Tape(bool persistent, bool watch_accessed_variables)
        {
            _persistent = persistent;
            _created_eagerly = tf.Context.executing_eagerly();
            tensor_tape_ = new TensorTape();
            op_tape_ = new OpTape();
            tensor_usage_ = new UnorderedMap<long, long>();
            if(_created_eagerly)
                tf.Context.start_step();
            // nesting_id = ++tape_nesting_id_counter;
        }

        /// <summary>
        /// Marks this tensor to be watched by the given tape.
        /// </summary>
        /// <param name="x"></param>
        public void Watch(Tensor x)
        {
            tf.Logger.Debug($"Watch tensor id={x.Id}, name={x.name}");
            tensor_tape_.emplace(x.Id, -1);
        }

        public bool ShouldRecord(long[] tensor_ids, TF_DataType[] tensor_dtypes)
        {
            Debug.Assert(tensor_ids.Length == tensor_dtypes.Length);
            for (int i = 0; i < tensor_ids.Length; ++i)
            {
                if (tensor_tape_.find(tensor_ids[i]) && IsDtypeTrainable(tensor_dtypes[i]))
                {
                    return true;
                }
            }
            return false;
        }

        public void VariableAccessed(IVariableV1 variable)
        {
            Watch(variable.Handle);
        }

        public IVariableV1[] WatchedVariables()
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

        public void StartRecord()
        {
            if (_recording)
                throw new ValueError("Tape is still recording, This can happen if you try to " +
                    "re-enter an already-active tape.");
            _recording = true;
        }

        public void StopRecord()
        {
            if (!_recording)
                throw new ValueError("Tape is not recording.");
            if (_created_eagerly)
                tf.Context.end_step();
            _recording = false;
        }

        public void SetTapeId(int id)
        {
            _id = id;
        }

        public override string ToString()
            => $"Tape {_id} {(_recording ? "Recording" : "Stopped")}";
    }
}
