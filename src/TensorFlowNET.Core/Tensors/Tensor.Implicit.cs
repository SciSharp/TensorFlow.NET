using NumSharp;
using System;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static implicit operator IntPtr(Tensor tensor)
        {
            return tensor._handle;
        }

        public static implicit operator Operation(Tensor tensor)
            => tensor?.op;

        public static implicit operator TF_Tensor(Tensor tensor)
            => new TF_Tensor(tensor._handle);

        public static implicit operator Tensor(IntPtr handle)
            => new Tensor(handle);

        public static implicit operator Tensor(NDArray nd)
            => tf.convert_to_tensor(nd);
    }
}
