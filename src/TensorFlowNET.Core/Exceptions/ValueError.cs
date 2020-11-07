namespace Tensorflow
{
    public class ValueError : TensorflowException
    {
        public ValueError() : base()
        {

        }

        public ValueError(string message) : base(message)
        {

        }
    }
}
