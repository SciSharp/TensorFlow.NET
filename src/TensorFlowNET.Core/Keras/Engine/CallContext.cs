namespace Tensorflow.Keras.Engine
{
    public class CallContext
    {
        public CallContextManager enter()
        {
            return new CallContextManager();
        }
    }
}
