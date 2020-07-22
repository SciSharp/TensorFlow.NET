using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor : Tensor
    {
        public override string Device => c_api.StringPiece(c_api.TFE_TensorHandleDeviceName(EagerTensorHandle, tf.status.Handle));

        public override int rank => c_api.TFE_TensorHandleNumDims(EagerTensorHandle, tf.status.Handle);

        public static int GetRank(IntPtr handle)
        {
            var tfe_tensor_handle = c_api.TFE_EagerTensorHandle(handle);
            return c_api.TFE_TensorHandleNumDims(tfe_tensor_handle, tf.status.Handle);
        }

        public static int[] GetDims(IntPtr handle)
        {
            var tfe_tensor_handle = c_api.TFE_EagerTensorHandle(handle);
            var dims = new int[c_api.TFE_TensorHandleNumDims(tfe_tensor_handle, tf.status.Handle)];
            for (int i = 0; i < dims.Length; i++)
                dims[i] = c_api.TFE_TensorHandleDim(tfe_tensor_handle, i, tf.status.Handle);
            return dims;
        }
    }
}
