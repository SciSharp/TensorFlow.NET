using System.Threading;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class ops
    {
        private static readonly ThreadLocal<DefaultGraphStack> _defaultGraphFactory = new ThreadLocal<DefaultGraphStack>(() => new DefaultGraphStack());
        private static volatile Session _singleSesson;
        private static volatile DefaultGraphStack _singleGraphStack;
        private static readonly object _threadingLock = new object();

        public static DefaultGraphStack default_graph_stack
        {
            get
            {
                if (!isSingleThreaded)
                    return _defaultGraphFactory.Value;

                if (_singleGraphStack == null)
                {
                    lock (_threadingLock)
                    {
                        if (_singleGraphStack == null)
                            _singleGraphStack = new DefaultGraphStack();
                    }
                }

                return _singleGraphStack;
            }
        }

        private static bool isSingleThreaded = false;

        /// <summary>
        ///     Does this library ignore different thread accessing.
        /// </summary>
        /// <remarks>https://github.com/SciSharp/TensorFlow.NET/wiki/Multithreading </remarks>
        public static bool IsSingleThreaded
        {
            get => isSingleThreaded;
            set
            {
                if (value)
                    enforce_singlethreading();
                else
                    enforce_multithreading();
            }
        }

        /// <summary>
        ///     Forces the library to ignore different thread accessing.
        /// </summary>
        /// <remarks>https://github.com/SciSharp/TensorFlow.NET/wiki/Multithreading <br></br>Note that this discards any sessions and graphs used in a multithreaded manner</remarks>
        public static void enforce_singlethreading()
        {
            isSingleThreaded = true;
        }

        /// <summary>
        ///     Forces the library to provide a separate <see cref="Session"/> and <see cref="Graph"/> to every different thread accessing.
        /// </summary>
        /// <remarks>https://github.com/SciSharp/TensorFlow.NET/wiki/Multithreading <br></br>Note that this discards any sessions and graphs used in a singlethreaded manner</remarks>
        public static void enforce_multithreading()
        {
            isSingleThreaded = false;
        }

        /// <summary>
        /// Returns the default session for the current thread.
        /// </summary>
        /// <returns>The default `Session` being used in the current thread.</returns>
        public static Session get_default_session()
        {
            if (!isSingleThreaded)
                return tf.defaultSession;

            if (_singleSesson == null)
            {
                lock (_threadingLock)
                {
                    if (_singleSesson == null)
                        _singleSesson = new Session();
                }
            }

            return _singleSesson;
        }

        /// <summary>
        /// Returns the default session for the current thread.
        /// </summary>
        /// <returns>The default `Session` being used in the current thread.</returns>
        public static Session set_default_session(Session sess)
        {
            if (!isSingleThreaded)
                return tf.defaultSession = sess;

            lock (_threadingLock)
            {
                _singleSesson = sess;
            }

            return _singleSesson;
        }

        /// <summary>                                                                  
        ///     Returns the default graph for the current thread.                      
        ///                                                                            
        ///     The returned graph will be the innermost graph on which a              
        ///     `Graph.as_default()` context has been entered, or a global default     
        ///     graph if none has been explicitly created.                             
        ///                                                                            
        ///     NOTE: The default graph is a property of the current thread.If you     
        ///     create a new thread, and wish to use the default graph in that         
        ///     thread, you must explicitly add a `with g.as_default():` in that       
        ///     thread's function.
        /// </summary>
        /// <returns></returns>
        public static Graph get_default_graph()
            => default_graph_stack.get_default();

        public static Graph set_default_graph(Graph g)
            => default_graph_stack.get_controller(g);

        /// <summary>
        ///     Clears the default graph stack and resets the global default graph.
        ///     
        ///     NOTE: The default graph is a property of the current thread.This
        ///     function applies only to the current thread.Calling this function while
        ///     a `tf.Session` or `tf.InteractiveSession` is active will result in undefined
        ///     behavior. Using any previously created `tf.Operation` or `tf.Tensor` objects
        ///     after calling this function will result in undefined behavior.
        /// </summary>
        /// <returns></returns>
        public static void reset_default_graph()
        {
            //if (!_default_graph_stack.is_cleared())
            //    throw new InvalidOperationException("Do not use tf.reset_default_graph() to clear " +
            //                                    "nested graphs. If you need a cleared graph, " +
            //                                    "exit the nesting and create a new graph.");
            default_graph_stack.reset();
        }

        public static Graph peak_default_graph()
            => default_graph_stack.peak_controller();

        public static void pop_graph()
            => default_graph_stack.pop();
    }
}