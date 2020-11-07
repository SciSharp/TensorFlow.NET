namespace Tensorflow
{
    public partial class Tensor
    {
        public object[] Flatten()
        {
            return new Tensor[] { this };
        }
    }
}
