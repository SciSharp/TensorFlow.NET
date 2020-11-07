namespace Tensorflow
{
    public partial class Tensor
    {
        public Tensor Pack(object[] sequences)
        {
            return sequences[0] as Tensor;
        }
    }
}
