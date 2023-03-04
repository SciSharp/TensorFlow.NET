using Tensorflow.NumPy;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Tensorflow
{
    /// <summary>
    /// Tensors is used to represent a Tensor or a array of Tensor.
    /// It will simplify the API interface, it converts Tensor
    /// and Tensor[] to Tensors implicitily. And parse back to Tensor
    /// and Tensor[] from Tensors implicitily. 
    /// It works for tuple and scalar as well.
    /// </summary>
    public class Tensors : IEnumerable<Tensor>, IDisposable
    {
        List<Tensor> items = new List<Tensor>();

        public TF_DataType dtype => items.First().dtype;
        public Shape shape => items.First().shape;
        public int rank => items.First().rank;
        public Graph graph => items.First().graph;
        public bool IsList { get; set; }
        public int Length => items.Count();

        public Tensor this[int index]
        {
            get => items[index];
            set => items[index] = value;
        }

        public Tensor this[params string[] slices]
            => items.First()[slices];
        public Tensors(params Tensor[] tensors)
        {
            items.AddRange(tensors);
        }

        public Tensors(IEnumerable<Tensor> tensors)
        {
            items.AddRange(tensors);
        }

        public Tensors(NDArray nd)
        {
            items.Add(ops.convert_to_tensor(nd));
        }

        public IEnumerator<Tensor> GetEnumerator()
        {
            foreach (var tensor in items)
                yield return tensor;
        }

        public void Add(Tensor tensor)
            => items.Add(tensor);

        public void AddRange(Tensor[] tensors)
            => items.AddRange(tensors);

        public void Insert(int index, Tensor tensor)
            => items.Insert(index, tensor);

        IEnumerator IEnumerable.GetEnumerator()
            => GetEnumerator();

        public NDArray numpy()
        {
            EnsureSingleTensor(this, "nnumpy");
            return this[0].numpy();
        }

        public T[] ToArray<T>() where T: unmanaged
        {
            EnsureSingleTensor(this, $"ToArray<{typeof(T)}>");
            return this[0].ToArray<T>();
        }

        #region Explicit Conversions
        public unsafe static explicit operator bool(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to bool");
            return (bool)tensor[0];
        }

        public unsafe static explicit operator sbyte(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to sbyte");
            return (sbyte)tensor[0];
        }

        public unsafe static explicit operator byte(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to byte");
            return (byte)tensor[0];
        }

        public unsafe static explicit operator ushort(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to ushort");
            return (ushort)tensor[0];
        }

        public unsafe static explicit operator short(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to short");
            return (short)tensor[0];
        }

        public unsafe static explicit operator int(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to int");
            return (int)tensor[0];
        }

        public unsafe static explicit operator uint(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to uint");
            return (uint)tensor[0];
        }

        public unsafe static explicit operator long(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to long");
            return (long)tensor[0];
        }

        public unsafe static explicit operator ulong(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to ulong");
            return (ulong)tensor[0];
        }

        public unsafe static explicit operator float(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to byte");
            return (byte)tensor[0];
        }

        public unsafe static explicit operator double(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to double");
            return (double)tensor[0];
        }

        public unsafe static explicit operator string(Tensors tensor)
        {
            EnsureSingleTensor(tensor, "explicit conversion to string");
            return (string)tensor[0];
        }
        #endregion

        #region Implicit Conversions
        public static implicit operator Tensors(Tensor tensor)
            => new Tensors(tensor);

        public static implicit operator Tensors((Tensor, Tensor) tuple)
            => new Tensors(tuple.Item1, tuple.Item2);

        [AutoNumPy]
        public static implicit operator Tensors(NDArray nd)
            => new Tensors(nd);

        public static implicit operator Tensors(Tensor[] tensors)
            => new Tensors(tensors);

        public static implicit operator Tensors(List<Tensor> tensors)
            => new Tensors(tensors.ToArray());

        public static implicit operator Tensor(Tensors tensors)
            => tensors.FirstOrDefault();

        public static implicit operator Tensor[](Tensors tensors)
            => tensors.items.ToArray();

        #endregion

        public void Deconstruct(out Tensor a, out Tensor b)
        {
            a = items[0];
            b = items[1];
        }

        private static void EnsureSingleTensor(Tensors tensors, string methodnName)
        {
            if(tensors.Length == 0)
            {
                throw new ValueError($"Method `{methodnName}` of `Tensors` cannot be used when `Tensors` contains no Tensor.");
            }
            else if(tensors.Length > 1)
            {
                throw new ValueError($"Method `{methodnName}` of `Tensors` cannot be used when `Tensors` contains more than one Tensor.");
            }
        }

        public override string ToString()
            => items.Count() == 1
               ? items.First().ToString()
               : items.Count() + " Tensors" + ". " + string.Join(", ", items.Select(x => x.name));

        public void Dispose()
        {
            foreach (var item in items)
                item.Dispose();
        }
    }
}
