using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Common.Extensions
{
    public static class LinqExtensions
    {
#if NETSTANDARD2_0
        public static IEnumerable<T> TakeLast<T>(this IEnumerable<T> sequence, int count)
        {
            return sequence.Skip(sequence.Count() - count);
        }

        public static IEnumerable<T> SkipLast<T>(this IEnumerable<T> sequence, int count)
        {
            return sequence.Take(sequence.Count() - count);
        }
#endif
        public static Tensors ToTensors(this IEnumerable<Tensor> tensors)
        {
            return new Tensors(tensors);
        }
    }
}
