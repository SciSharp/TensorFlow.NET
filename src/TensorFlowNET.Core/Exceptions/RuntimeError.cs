namespace Tensorflow
{
    public class RuntimeError : TensorflowException
    {
        public RuntimeError() : base()
        {

        }

        public RuntimeError(string message) : base(message)
        {

        }
    }
}
