using System.Linq;
using Tensorflow.Eager;

namespace Tensorflow.Framework.Models
{
    public class TensorSpec : DenseSpec
    {
        public TensorSpec(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null) :
            base(shape, dtype, name)
        {
            
        }

        public TensorSpec _unbatch()
        {
            if (_shape.ndim == 0)
                throw new ValueError("Unbatching a tensor is only supported for rank >= 1");

            return new TensorSpec(_shape.dims.Skip(1).ToArray(), _dtype);
        }

        public TensorSpec _batch(int dim = -1)
        {
            var shapes = shape.dims.ToList();
            shapes.Insert(0, dim);
            return new TensorSpec(shapes.ToArray(), _dtype);
        }

        public static TensorSpec FromTensor(Tensor tensor, string? name = null)
        {
            if(tensor is EagerTensor)
            {
                return new TensorSpec(tensor.shape, tensor.dtype, name);
            }
            else
            {
                return new TensorSpec(tensor.shape, tensor.dtype, name ?? tensor.name);
            }
        }
    }
}
