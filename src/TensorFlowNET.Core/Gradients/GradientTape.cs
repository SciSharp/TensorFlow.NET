using System;
using System.Collections.Generic;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Gradients
{
    /// <summary>
    /// Gradient Tape Set
    /// Record operations for automatic differentiation.
    /// 
    /// Operations are recorded if they are executed within this context manager and
    /// at least one of their inputs is being "watched".
    /// 
    /// Trainable variables (created by `tf.Variable` or `tf.compat.v1.get_variable`,
    /// where `trainable=True` is default in both cases) are automatically watched.
    /// Tensors can be manually watched by invoking the `watch` method on this context
    /// manager.
    /// </summary>
    public class GradientTape : IDisposable
    {
        int _nextTapeId;
        ITape _tape => _tapeSet.Peek();
        Stack<ITape> _tapeSet;

        public GradientTape()
        {
            _tapeSet = new Stack<ITape>();
        }

        /// <summary>
        /// New tape onto the tape stack.
        /// </summary>
        public ITape PushTape(bool persistent = false,
            bool watch_accessed_variables = true)
        {
            // Enters a context inside which operations are recorded on this tape.
            if (tf.Context.executing_eagerly())
                tf.Context.ensure_initialized();

            var tape = new Tape(persistent, watch_accessed_variables);
            tape.SetTapeId(_nextTapeId++);
            _tapeSet.Push(tape);
            return tape;
        }

        public void PushTape(ITape tape)
        {
            // Enters a context inside which operations are recorded on this tape.
            if (tf.Context.executing_eagerly())
                tf.Context.ensure_initialized();

            _tapeSet.Push(tape);
        }

        ITape PopTape()
        {
            _tape.StopRecord();
            return _tapeSet.Pop();
        }

        /// <summary>
        /// Marks this tensor to be watched by the given tape.
        /// </summary>
        /// <param name="x"></param>
        public void watch(Tensor x)
        {
            if (!_tapeSet.Any())
                return;
            _tape.Watch(x);
        }

        /// <summary>
        /// Computes the gradient using operations recorded in context of this tape.
        /// </summary>
        /// <param name="target"></param>
        /// <param name="source"></param>
        /// <returns></returns>
        public Tensor gradient(Tensor target, Tensor source, List<Tensor> output_gradients = null, 
            string unconnected_gradients = null)
        {
            if(_tape is null)
            {
                throw new RuntimeError("A non-persistent GradientTape can only be used to " +
                    "compute one set of gradients (or jacobians).");
            }
            
            ITape tape = stop_recording();

            var results = tf.Runner.TFE_TapeGradient(tape,
                new[] { target },
                new[] { source },
                output_gradients,
                new[] { source }, 
                unconnected_gradients);

            return results[0];
        }

        public Tensor gradient(Tensor target, ResourceVariable source, List<Tensor> output_gradients = null,
            string unconnected_gradients = null)
        {
            var results = gradient(target, new List<IVariableV1> { source }, output_gradients, unconnected_gradients);

            return results[0];
        }

        public (Tensor, Tensor) gradient(Tensor target, (ResourceVariable, ResourceVariable) sources, List<Tensor> output_gradients = null,
            string unconnected_gradients = null)
        {
            var results = gradient(target, new List<IVariableV1> { sources.Item1, sources.Item2 }, output_gradients, unconnected_gradients);

            return (results[0], results[1]);
        }

        public Tensor[] gradient(Tensor target, IEnumerable<IVariableV1> sources, List<Tensor> output_gradients = null,
            string unconnected_gradients = null)
        {
            if (_tape is null)
            {
                throw new RuntimeError("A non-persistent GradientTape can only be used to " +
                    "compute one set of gradients (or jacobians).");
            }
            var tape = stop_recording();

            var results = tf.Runner.TFE_TapeGradient(tape,
                new[] { target },
                sources.Select(x => x.Handle).ToArray(),
                output_gradients,
                sources.Select(x => x.Handle).ToArray(), 
                unconnected_gradients);

            if (!tape.Persistent)
            {
                // Keep track of watched variables before setting tape to None
                // _watched_variables = _tape.WatchedVariables();
            }

            return results;
        }

        /// <summary>
        /// Temporarily stops recording operations on this tape.
        /// </summary>
        public ITape stop_recording()
        {
            var tape = _tape;
            if (!tape.Persistent)
                tape = PopTape();
            return tape;
        }

        public Stack<ITape> GetTapeSet()
            => _tapeSet;

        public void Dispose()
        {
            _tapeSet.Clear();
        }
    }
}
