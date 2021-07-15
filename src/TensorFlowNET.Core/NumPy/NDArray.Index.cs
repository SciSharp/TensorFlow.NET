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
            get => _tensor[index.Select(x => new Slice
            {
                Start = x,
                Stop = x + 1,
                IsIndex = true
            }).ToArray()];

            set => SetData(index.Select(x => new Slice
            {
                Start = x,
                Stop = x + 1,
                IsIndex = true
            }), value);
        }

        public NDArray this[params Slice[] slices]
        {
            get => _tensor[slices];
            set => SetData(slices, value);
        }

        public NDArray this[NDArray mask]
        {
            get
            {
                throw new NotImplementedException("");
            }

            set
            {
                throw new NotImplementedException("");
            }
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
                    throw new NotImplementedException("");

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
