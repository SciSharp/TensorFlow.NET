using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow.Gradients
{
    public class Tape : DisposableObject
    {
        public GradientTape tape { get; set; }
        public int nesting_id { get; set; }

        public Tape(bool persistent, bool watch_accessed_variables)
        {
            _handle = c_api.TFE_TapeSetNew(persistent, watch_accessed_variables);
        }

        public void watch(EagerTensor x)
        {
            c_api.TFE_TapeWatch(_handle, x.EagerTensorHandle);
        }

        public void pop_tape(Tape tape)
        {
            c_api.TFE_TapeSetRemove(tape);
        }

        public static bool IsDtypeTrainable(DataType dtype)
        {
            switch (dtype)
            {
                case DataType.DtHalf:
                case DataType.DtBfloat16:
                case DataType.DtFloat:
                case DataType.DtDouble:
                case DataType.DtComplex64:
                case DataType.DtComplex128:
                case DataType.DtResource:
                case DataType.DtVariant:
                    return true;
                default:
                    return false;
            }
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
        }

        public static implicit operator IntPtr(Tape tape)
            => tape._handle;
    }
}
