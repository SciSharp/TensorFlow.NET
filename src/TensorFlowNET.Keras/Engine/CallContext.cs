using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public class CallContext
    {
        public CallContextManager enter(bool build_graph)
        {
            return new CallContextManager(build_graph);
        }
    }
}
