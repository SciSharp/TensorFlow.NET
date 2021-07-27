using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.NumPy
{
    public class ShapeHelper
    {
        public static long GetSize(Shape shape)
        {
            // scalar
            if (shape.ndim == 0)
                return 1;

            var computed = 1L;
            for (int i = 0; i < shape.ndim; i++)
            {
                var val = shape.dims[i];
                if (val == 0)
                    return 0;
                else if (val < 0)
                    continue;
                computed *= val;
            }

            return computed;
        }

        public static long[] GetStrides(Shape shape)
        {
            var strides = new long[shape.ndim];

            if (shape.ndim == 0)
                return strides;
            
            strides[strides.Length - 1] = 1;
            for (int idx = strides.Length - 1; idx >= 1; idx--)
                strides[idx - 1] = strides[idx] * shape.dims[idx];

            return strides;
        }

        public static Shape GetShape(Shape shape1, params Slice[] slices)
        {
            var new_dims = shape1.dims.ToArray();
            slices = SliceHelper.AlignWithShape(shape1, slices);

            for (int i = 0; i < shape1.dims.Length; i++)
            {
                Slice slice = slices[i];
                if (slice.Equals(Slice.All))
                    new_dims[i] = shape1.dims[i];
                else if (slice.IsIndex)
                    new_dims[i] = 1;
                else // range
                    new_dims[i] = (slice.Stop ?? shape1.dims[i]) - (slice.Start ?? 0);
            }

            // strip first dim if is index
            var return_dims = new List<long>();
            for (int i = 0; i< new_dims.Length; i++)
            {
                if (slices[i].IsIndex)
                    continue;
                return_dims.add(new_dims[i]);
            }

            return new Shape(return_dims.ToArray());
        }

        public static bool Equals(Shape shape, object target)
        {
            switch (target)
            {
                case Shape shape1:
                    if (shape.ndim == -1 && shape1.ndim == -1)
                        return false;
                    else if (shape.ndim != shape1.ndim)
                        return false;
                    return Enumerable.SequenceEqual(shape1.dims, shape.dims);
                case long[] shape2:
                    if (shape.ndim != shape2.Length)
                        return false;
                    return Enumerable.SequenceEqual(shape.dims, shape2);
                default:
                    return false;
            }
        }

        public static string ToString(Shape shape)
        {
            return shape.ndim switch
            {
                -1 => "<unknown>",
                0 => "()",
                1 => $"({shape.dims[0].ToString().Replace("-1", "None")},)",
                _ => $"({string.Join(", ", shape.dims).Replace("-1", "None")})"
            };
        }

        public static long GetOffset(Shape shape, params int[] indices)
        {
            if (shape.ndim == 0 && indices.Length == 1)
                return indices[0];

            long offset = 0;
            var strides = shape.strides;
            for (int i = 0; i < indices.Length; i++)
                offset += strides[i] * indices[i];

            if (offset < 0)
                throw new NotImplementedException("");

            return offset;
        }
    }
}
