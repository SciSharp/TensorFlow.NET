using NumSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Gradients;

namespace Tensorflow
{
    /// <summary>
    /// Tensors is used to represent a Tensor or a array of Tensor.
    /// It will simplify the API interface, it converts Tensor
    /// and Tensor[] to Tensors implicitily. And parse back to Tensor
    /// and Tensor[] from Tensors implicitily. 
    /// It works for tuple and scalar as well.
    /// </summary>
    public class Tensors : IEnumerable<Tensor>
    {
        Tensor[] items;

        public TF_DataType dtype => items.First().dtype;
        public TensorShape shape => items.First().TensorShape;
        public int rank => items.First().rank;
        public Graph graph => items.First().graph;
        public bool IsEagerTensor => items.First().IsEagerTensor;

        public Tensor this[int index] => items[index];

        public Tensors(params Tensor[] tensors)
        {
            items = tensors;
        }

        public Tensors(NDArray nd)
        {
            items = new[] { ops.convert_to_tensor(nd) };
        }

        public IEnumerator<Tensor> GetEnumerator()
        {
            foreach (var tensor in items)
                yield return tensor;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            throw new NotImplementedException();
        }

        public static implicit operator Tensors(Tensor tensor)
            => new Tensors(tensor);

        public static implicit operator Tensors(NDArray nd)
            => new Tensors(nd);

        public static implicit operator Tensors(Tensor[] tensors)
            => new Tensors(tensors);

        public static implicit operator Tensors(List<Tensor> tensors)
            => new Tensors(tensors.ToArray());

        public static implicit operator Tensor(Tensors tensors)
            => tensors.FirstOrDefault();

        public static implicit operator Tensor[](Tensors tensors)
            => tensors.items;

        public override string ToString()
            => items.Length == 1
               ? items.First().ToString()
               : items.Length + " Tensors" + ". " + string.Join(", ", items.Select(x => x.name));
    }
}
