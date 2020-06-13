using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Tensorflow
{
    public abstract class PointerInputs<T>
        where T : IPointerInputs, new()
    {
        protected T[] data;
        public int Length
            => data.Length;

        public IntPtr[] Points
            => data.Select(x => x.ToPointer()).ToArray();

        public PointerInputs(params T[] data)
            => this.data = data;

        public T this[int idx] 
            => data[idx];

        public T[] Items
            => data;

        public static implicit operator IntPtr[](PointerInputs<T> inputs)
            => inputs.data.Select(x => x.ToPointer()).ToArray();
    }
}
