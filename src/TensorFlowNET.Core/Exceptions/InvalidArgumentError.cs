namespace Tensorflow
{
    public class InvalidArgumentError : TensorflowException
    {
        public InvalidArgumentError() : base()
        {

        }

        public InvalidArgumentError(string message) : base(message)
        {

        }
    }
}
