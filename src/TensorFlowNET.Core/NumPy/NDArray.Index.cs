using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NDArray
    {
        public NDArray this[params int[] indices]
        {
            get => GetData(indices.Select(x => new Slice
            {
                Start = x,
                Stop = x + 1,
                IsIndex = true
            }).ToArray());

            set => SetData(indices.Select(x => 
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
                if (mask.dtype == TF_DataType.TF_BOOL)
                    return GetData(enumerate(mask.ToArray<bool>()).Where(x => x.Item2).Select(x => x.Item1).ToArray());
                else if (mask.dtype == TF_DataType.TF_INT32)
                    return GetData(mask.ToArray<int>());
                else if (mask.dtype == TF_DataType.TF_INT64)
                    return GetData(mask.ToArray<long>().Select(x => Convert.ToInt32(x)).ToArray());
                else if (mask.dtype == TF_DataType.TF_FLOAT)
                    return GetData(mask.ToArray<float>().Select(x => Convert.ToInt32(x)).ToArray());

                throw new NotImplementedException("");
            }

            set
            {
                if (mask.dtype == TF_DataType.TF_BOOL)
                    MaskData(mask, value);
                else
                    throw new NotImplementedException("");
            }
        }

        [AutoNumPy]
        unsafe NDArray GetData(Slice[] slices)
        {
            if (shape.IsScalar)
                return GetScalar();

            if (SliceHelper.AreAllIndex(slices, out var indices1))
            {
                var newshape = ShapeHelper.GetShape(shape, slices);
                if (newshape.IsScalar)
                {
                    var offset = ShapeHelper.GetOffset(shape, indices1);
                    return GetScalar((ulong)offset);
                }
                else
                {
                    return GetArrayData(newshape, indices1);
                }
            }
            else if (slices.Count() == 1)
            {
                var slice = slices[0];
                if (slice.Step == 1)
                {
                    var newshape = ShapeHelper.GetShape(shape, slice);
                    var array = new NDArray(newshape, dtype: dtype);

                    var new_dims = new int[shape.ndim];
                    new_dims[0] = slice.Start ?? 0;
                    //for (int i = 1; i < shape.ndim; i++)
                        //new_dims[i] = (int)shape.dims[i];

                    var offset = ShapeHelper.GetOffset(shape, new_dims);
                    var src = (byte*)data + (ulong)offset * dtypesize;
                    var dst = (byte*)array.data;
                    var len = (ulong)newshape.size * dtypesize;

                    System.Buffer.MemoryCopy(src, dst, len, len);

                    return array;
                }
            }

            // default, performance is bad
            var tensor = base[slices.ToArray()];
            if (tensor.Handle == null)
            {
                if (tf.executing_eagerly())
                    tensor = tf.get_default_session().eval(tensor);
            }

            return new NDArray(tensor, tf.executing_eagerly());
        }

        unsafe T GetAtIndex<T>(params int[] indices) where T : unmanaged
        {
            var offset = (ulong)ShapeHelper.GetOffset(shape, indices);
            return *((T*)data + offset);
        }

        unsafe NDArray GetScalar(ulong offset = 0)
        {
            var array = new NDArray(Shape.Scalar, dtype: dtype);
            var src = (byte*)data + offset * dtypesize;
            System.Buffer.MemoryCopy(src, array.buffer.ToPointer(), dtypesize, dtypesize);
            return array;
        }

        unsafe NDArray GetArrayData(Shape newshape, int[] indices)
        {
            var offset = ShapeHelper.GetOffset(shape, indices);
            var len = (ulong)newshape.size * dtypesize;
            var array = new NDArray(newshape, dtype: dtype);

            var src = (byte*)data + (ulong)offset * dtypesize;
            System.Buffer.MemoryCopy(src, array.data.ToPointer(), len, len);

            return array;
        }

        unsafe NDArray GetData(int[] indices, int axis = 0)
        {
            if (shape.IsScalar)
                return GetScalar();

            if(axis == 0)
            {
                var dims = shape.as_int_list();
                dims[0] = indices.Length;

                var array = np.ndarray(dims, dtype: dtype);

                dims[0] = 1;
                var len = new Shape(dims).size * dtype.get_datatype_size();

                int dst_index = 0;
                foreach (var pos in indices)
                {
                    var src_offset = (ulong)ShapeHelper.GetOffset(shape, pos);
                    var dst_offset = (ulong)ShapeHelper.GetOffset(array.shape, dst_index++);

                    var src = (byte*)data + src_offset * dtypesize;
                    var dst = (byte*)array.data + dst_offset * dtypesize;
                    System.Buffer.MemoryCopy(src, dst, len, len);
                }

                return array;
            }
            else
                throw new NotImplementedException("");
        }

        void SetData(IEnumerable<Slice> slices, NDArray array)
            => SetData(array, slices.ToArray(), new int[shape.ndim].ToArray(), -1);

        unsafe void SetData(NDArray src, Slice[] slices, int[] indices, int currentNDim)
        {
            if (dtype != src.dtype)
                // src = src.astype(dtype);
                throw new ArrayTypeMismatchException($"Required dtype {dtype} but {src.dtype} is assigned.");

            if (!slices.Any())
                return;

            if (shape.Equals(src.shape))
            {
                System.Buffer.MemoryCopy(src.data.ToPointer(), data.ToPointer(), src.bytesize, src.bytesize);
                return;
            }

            // first iteration
            if(currentNDim == -1)
            {
                slices = SliceHelper.AlignWithShape(shape, slices);
            }

            // last dimension
            if (currentNDim == ndim - 1)
            {
                var offset = (int)ShapeHelper.GetOffset(shape, indices);
                var dst = data + offset * (int)dtypesize;
                System.Buffer.MemoryCopy(src.data.ToPointer(), dst.ToPointer(), src.bytesize, src.bytesize);
                return;
            }

            currentNDim++;
            var slice = slices[currentNDim];
            
            var start = slice.Start ?? 0;
            var stop = slice.Stop ?? (int)dims[currentNDim];
            var step = slice.Step;

            if(step != 1)
            {
                for (var i = start; i < stop; i += step)
                {
                    if (i >= dims[currentNDim])
                        throw new OutOfRangeError($"Index should be in [0, {dims[currentNDim]}] but got {i}");

                    indices[currentNDim] = i;
                    if (currentNDim < ndim - src.ndim)
                    {
                        SetData(src, slices, indices, currentNDim);
                    }
                    else
                    {
                        var srcIndex = (i - start) / step;
                        SetData(src[srcIndex], slices, indices, currentNDim);
                    }
                }
            }
            else
            {
                for (var i = start; i < stop; i++)
                {
                    if (i >= dims[currentNDim])
                        throw new OutOfRangeError($"Index should be in [0, {dims[currentNDim]}] but got {i}");

                    indices[currentNDim] = i;
                    if (currentNDim < ndim - src.ndim)
                    {
                        SetData(src, slices, indices, currentNDim);
                    }
                    // last dimension
                    else if(currentNDim == ndim - 1)
                    {
                        SetData(src, slices, indices, currentNDim);
                        break;
                    }
                    else if(SliceHelper.IsContinuousBlock(slices, currentNDim))
                    {
                        var offset = (int)ShapeHelper.GetOffset(shape, indices);
                        var dst = data + offset * (int)dtypesize;
                        System.Buffer.MemoryCopy(src.data.ToPointer(), dst.ToPointer(), src.bytesize, src.bytesize);
                        return;
                    }
                    else
                    {
                        var srcIndex = i - start;
                        SetData(src[srcIndex], slices, indices, currentNDim);
                    }
                }
            }

            // reset indices
            indices[currentNDim] = 0;
        }

        unsafe void MaskData(NDArray mask, NDArray value)
        {
            var masks = mask.ToArray<bool>();
            var s1 = new Shape(dims.Skip(mask.rank).ToArray());
            var val = tf.fill(s1, value).numpy();
            for (int i = 0; i < masks.Length; i++)
            {
                if (masks[i])
                    this[i] = val;
            }
        }
    }
}
