using Tensorflow.Gradients;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        TapeTensor TapeTensorFromTensor(Tensor tensor)
        {
            return new TapeTensor(tensor.Id, tensor.dtype, tensor.shape);
        }
    }
}
