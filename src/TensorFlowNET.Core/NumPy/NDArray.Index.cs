using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public NDArray this[params int[] index]
        {
            get => GetData(index.Select(x => new Slice
            {
                Start = x,
                Stop = x + 1,
                IsIndex = true
            }));

            set => SetData(index.Select(x => 
            {
                if(x < 0)
                    x = (int)dims[0] + x;
                
                var slice = new Slice
                {
                    Start = x,
                    Stop = x + 1,
                    IsIndex = true
                };

                return slice;
            }), value);
        }

        public NDArray this[params Slice[] slices]
        {
            get => GetData(slices);
            set => SetData(slices, value);
        }

        public NDArray this[NDArray mask]
        {
            get
            {
                if(mask.dtype == TF_DataType.TF_INT32)
                    return GetData(mask.ToArray<int>());

                throw new NotImplementedException("");
            }

            set
            {
                throw new NotImplementedException("");
            }
        }

        NDArray GetData(IEnumerable<Slice> slices)
        {
            var tensor = _tensor[slices.ToArray()];
            return new NDArray(tensor);
        }

        NDArray GetData(int[] indices, int axis = 0)
        {
            if(axis == 0)
            {
                var dims = shape.as_int_list();
                dims[0] = indices.Length;

                var array = np.ndarray(dims, dtype: dtype);

                dims[0] = 1;
                var bytesize = new Shape(dims).size * dtype.get_datatype_size();

                int dst_index = 0;
                foreach (var index in indices)
                {
                    var src_offset = (ulong)ShapeHelper.GetOffset(shape, index);
                    var dst_offset = (ulong)ShapeHelper.GetOffset(array.shape, dst_index++);
                    unsafe
                    {
                        var src = (byte*)data + src_offset * dtypesize;
                        var dst = (byte*)array.data.ToPointer() + dst_offset * dtypesize;
                        System.Buffer.MemoryCopy(src, dst, bytesize, bytesize);
                    }
                }

                return array;
            }
            else
                throw new NotImplementedException("");
        }

        void SetData(IEnumerable<Slice> slices, NDArray array)
            => SetData(slices, array, -1, slices.Select(x => 0).ToArray());

        void SetData(IEnumerable<Slice> slices, NDArray array, int currentNDim, int[] indices)
        {
            if (dtype != array.dtype)
                throw new ArrayTypeMismatchException($"Required dtype {dtype} but {array.dtype} is assigned.");

            if (!slices.Any())
                return;

            var slice = slices.First();

            if (slices.Count() == 1)
            {

                if (slice.Step != 1)
                    throw new NotImplementedException("slice.step should == 1");

                if (slice.Start < 0)
                    throw new NotImplementedException("slice.start should > -1");

                indices[indices.Length - 1] = slice.Start ?? 0;
                var offset = (ulong)ShapeHelper.GetOffset(shape, indices);
                var bytesize = array.bytesize;
                unsafe
                {
                    var dst = (byte*)data + offset * dtypesize;
                    System.Buffer.MemoryCopy(array.data.ToPointer(), dst, bytesize, bytesize);
                }

                return;
            }

            currentNDim++;
            for (var i = slice.Start ?? 0; i < slice.Stop; i++)
            {
                indices[currentNDim] = i;
                SetData(slices.Skip(1), array, currentNDim, indices);
            }
        }
    }
}
