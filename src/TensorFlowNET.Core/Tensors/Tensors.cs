using NumSharp;
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
        public TensorShape shape => items.First().TensorShape;
        public int rank => items.First().rank;
        public Graph graph => items.First().graph;
        public bool IsEagerTensor => items.First().IsEagerTensor;
        public bool IsList { get; set; }
        public int Length => items.Count();

        public Tensor this[int index]
        {
            get
            {
                return items[index];
            }

            set
            {
                items[index] = value;
            }
        }

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

        public static implicit operator Tensors(Tensor tensor)
            => new Tensors(tensor);

        public static implicit operator Tensors((Tensor, Tensor) tuple)
            => new Tensors(tuple.Item1, tuple.Item2);

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

        public void Deconstruct(out Tensor a, out Tensor b)
        {
            a = items[0];
            b = items[1];
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
