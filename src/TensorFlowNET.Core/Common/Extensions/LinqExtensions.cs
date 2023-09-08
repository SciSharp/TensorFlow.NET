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
        public static Tensors ToTensors(this Tensor[] tensors)
        {
            return new Tensors(tensors);
        }

        public static Tensors ToTensors(this IList<Tensor> tensors)
        {
            return new Tensors(tensors);
        }

        public static void Deconstruct<T1, T2, T3>(this (T1, T2, T3) values, out T1 first, out T2 second, out T3 third)
        {
            first = values.Item1;
            second = values.Item2;
            third = values.Item3;
        }
    }
}
