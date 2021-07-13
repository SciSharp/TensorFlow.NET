using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public NDArray this[int index]
        {
            get
            {
                return _tensor[index];
            }

            set
            {

            }
        }

        public NDArray this[params int[] index]
        {
            get
            {
                 return _tensor[index.Select(x => new Slice(x, x + 1)).ToArray()];
            }

            set
            {
                var offset = ShapeHelper.GetOffset(shape, index);
                unsafe
                {
                    if (dtype == TF_DataType.TF_BOOL)
                        *((bool*)data + offset) = value;
                    else if (dtype == TF_DataType.TF_UINT8)
                        *((byte*)data + offset) = value;
                    else if (dtype == TF_DataType.TF_INT32)
                        *((int*)data + offset) = value;
                    else if (dtype == TF_DataType.TF_INT64)
                        *((long*)data + offset) = value;
                    else if (dtype == TF_DataType.TF_FLOAT)
                        *((float*)data + offset) = value;
                    else if (dtype == TF_DataType.TF_DOUBLE)
                        *((double*)data + offset) = value;
                }
            }
        }

        public NDArray this[params Slice[] slices]
        {
            get
            {
                return _tensor[slices];
            }

            set
            {
                var pos = _tensor[slices];
                var len = value.bytesize;
                unsafe
                {
                    System.Buffer.MemoryCopy(value.data.ToPointer(), pos.TensorDataPointer.ToPointer(), len, len);
                }
                // _tensor[slices].assign(constant_op.constant(value));
            }
        }

        public NDArray this[NDArray mask]
        {
            get
            {
                throw new NotImplementedException("");
            }

            set
            {

            }
        }
    }
}
