using NumSharp;
using System;
using System.Linq;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor : Tensor
    {
        public override string Device
        {
            get
            {
                using var _ = EagerTensorHandle.Lease();
                return c_api.StringPiece(c_api.TFE_TensorHandleDeviceName(EagerTensorHandle, tf.Status.Handle));
            }
        }

        public override int rank => c_api.TFE_TensorHandleNumDims(EagerTensorHandle, tf.Status.Handle);

        public override void set_shape(TensorShape shape)
        {
            if (!shape.is_compatible_with(this.shape))
                throw new ValueError($"Tensor's shape is not compatible.");
        }

        public static int GetRank(IntPtr handle)
        {
            var tfe_tensor_handle = c_api.TFE_EagerTensorHandle(handle);
            return c_api.TFE_TensorHandleNumDims(tfe_tensor_handle, tf.Status.Handle);
        }

        public static int[] GetDims(IntPtr handle)
        {
            var tfe_tensor_handle = c_api.TFE_EagerTensorHandle(handle);
            var dims = new int[c_api.TFE_TensorHandleNumDims(tfe_tensor_handle, tf.Status.Handle)];
            for (int i = 0; i < dims.Length; i++)
                dims[i] = c_api.TFE_TensorHandleDim(tfe_tensor_handle, i, tf.Status.Handle);
            return dims;
        }
    }
}
