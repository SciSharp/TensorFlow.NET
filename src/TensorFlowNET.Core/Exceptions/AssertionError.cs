namespace Tensorflow.Exceptions;

public class AssertionError : TensorflowException
{
    public AssertionError() : base()
    {

    }

    public AssertionError(string message) : base(message)
    {

    }
}
