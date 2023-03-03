using System;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerTensor : Tensor
    {
        public override SafeTensorHandle Handle
        {
            get
            {
                Resolve();
                return _handle;
            }
        }

        public override IntPtr buffer
        {
            get
            {
                Resolve();
                return base.buffer;
            }
        }

        public override string Device => c_api.StringPiece(c_api.TFE_TensorHandleDeviceName(_eagerTensorHandle, tf.Status));
        public override TF_DataType dtype => c_api.TFE_TensorHandleDataType(_eagerTensorHandle);

        public override int rank => c_api.TFE_TensorHandleNumDims(EagerTensorHandle, tf.Status);

        public override ulong bytesize
        {
            get
            {
                Resolve();
                return base.bytesize;
            }
        }

        public override IntPtr TensorDataPointer
        {
            get
            {
                Resolve();
                return base.TensorDataPointer;
            }
        }

        protected override Shape GetShapeInternal()
        {
            var dims = new int[c_api.TFE_TensorHandleNumDims(_eagerTensorHandle, tf.Status)];
            for (int i = 0; i < dims.Length; i++)
                dims[i] = c_api.TFE_TensorHandleDim(_eagerTensorHandle, i, tf.Status);
            return dims;
        }

        protected override void SetShapeInternal(Shape value)
        {
            if (!shape.is_compatible_with(value))
                throw new ValueError($"Tensor's shape is not compatible.");
        }

        public static int GetRank(IntPtr handle)
        {
            var tfe_tensor_handle = c_api.TFE_EagerTensorHandle(handle);
            return c_api.TFE_TensorHandleNumDims(tfe_tensor_handle, tf.Status);
        }

        public static int[] GetDims(IntPtr handle)
        {
            var tfe_tensor_handle = c_api.TFE_EagerTensorHandle(handle);
            var dims = new int[c_api.TFE_TensorHandleNumDims(tfe_tensor_handle, tf.Status)];
            for (int i = 0; i < dims.Length; i++)
                dims[i] = c_api.TFE_TensorHandleDim(tfe_tensor_handle, i, tf.Status);
            return dims;
        }

        public override T[] ToArray<T>()
        {
            Resolve();
            return base.ToArray<T>();
        }
    }
}
