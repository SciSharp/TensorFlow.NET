using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public class CallContextManager : IDisposable
    {
        bool _build_graph;

        public CallContextManager(bool build_graph)
        {
            _build_graph = build_graph;
        }

        public void Dispose()
        {
            
        }
    }
}
