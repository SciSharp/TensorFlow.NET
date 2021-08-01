using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.NumPy
{
    public class SliceHelper
    {
        public static Slice[] AlignWithShape(Shape shape, Slice[] slices)
        {
            var ndim = shape.ndim;
            if (ndim == slices.Length)
                return slices;

            // align slices
            var new_slices = new List<Slice>();
            var slice_index = 0;

            for (int i = 0; i < ndim; i++)
            {
                if (slice_index > slices.Length - 1)
                {
                    new_slices.Add(Slice.All);
                    continue;
                }

                if (slices[slice_index] == Slice.All)
                {
                    new_slices.Add(Slice.All);
                    for (int j = 0; j < ndim - slices.Length; j++)
                    {
                        new_slices.Add(Slice.All);
                        i++;
                    }
                }
                else
                {
                    new_slices.Add(slices[slice_index]);
                }
                slice_index++;
            }

            return new_slices.ToArray();
        }

        public static bool AreAllIndex(Slice[] slices, out int[] indices)
        {
            indices = new int[slices.Length];
            for (int i = 0; i< slices.Length; i++)
            {
                indices[i] = slices[i].Start ?? 0;
                if (!slices[i].IsIndex)
                    return false;
            }
            return true;
        }

        public static bool IsContinuousBlock(Slice[] slices, int ndim)
        {
            for (int i = ndim + 1; i < slices.Length; i++)
            {
                if (slices[i].Equals(Slice.All))
                    continue;
                return false;
            }
            return true;
        }
    }
}
