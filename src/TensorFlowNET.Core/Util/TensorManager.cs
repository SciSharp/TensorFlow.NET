using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow
{
    public class TensorManager
    {
        Dictionary<IntPtr, EagerTensor> tensors;
        public TensorManager()
        {
            tensors = new Dictionary<IntPtr, EagerTensor>();
        }

        public EagerTensor GetTensor(IntPtr handle)
        {
            if (tensors.ContainsKey(handle))
                return tensors[handle];

            //return new EagerTensor(handle);
            tensors[handle] = new EagerTensor(handle);
            return tensors[handle];
        }

        public void Reset()
        {
            tensors.Clear();
        }
    }
}
