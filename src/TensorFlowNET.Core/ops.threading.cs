using System;
using System.Threading;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class ops
    {
        [ThreadStatic]
        static DefaultGraphStack default_graph_stack = new DefaultGraphStack();
        [ThreadStatic]
        static Session defaultSession;

        /// <summary>
        /// Returns the default session for the current thread.
        /// </summary>
        /// <returns>The default `Session` being used in the current thread.</returns>
        public static Session get_default_session()
        {
            if (defaultSession == null)
                defaultSession = new Session(tf.get_default_graph());

            return defaultSession;
        }

        /// <summary>
        /// Returns the default session for the current thread.
        /// </summary>
        /// <returns>The default `Session` being used in the current thread.</returns>
        public static Session set_default_session(Session sess)
        {
            defaultSession = sess;
            return sess;
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
        {
            if (default_graph_stack == null)
                default_graph_stack = new DefaultGraphStack();
            return default_graph_stack.get_default();
        }

        public static Graph set_default_graph(Graph g)
        {
            if (default_graph_stack == null)
                default_graph_stack = new DefaultGraphStack();
            return default_graph_stack.get_controller(g);
        }

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
            if (default_graph_stack == null)
                return;
            //if (!_default_graph_stack.is_cleared())
            //    throw new InvalidOperationException("Do not use tf.reset_default_graph() to clear " +
            //                                    "nested graphs. If you need a cleared graph, " +
            //                                    "exit the nesting and create a new graph.");
            default_graph_stack.reset();
        }

        public static Graph peak_default_graph()
        {
            if (default_graph_stack == null)
                default_graph_stack = new DefaultGraphStack();
            return default_graph_stack.peak_controller();
        }

        public static void pop_graph()
            => default_graph_stack.pop();
    }
}