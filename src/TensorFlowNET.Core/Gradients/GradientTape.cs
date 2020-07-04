using Google.Protobuf.WellKnownTypes;
using NumSharp.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow.Gradients
{
    /// <summary>
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
        bool _recording;
        public bool Recording => _recording;
        bool _persistent;
        bool _watch_accessed_variables;
        ResourceVariable[] _watched_variables;
        bool _created_eagerly;
        ITape _tape;

        public GradientTape(bool persistent = false,
            bool watch_accessed_variables = true)
        {
            _persistent = persistent;
            _watch_accessed_variables = watch_accessed_variables;
            _created_eagerly = tf.context.executing_eagerly();
            _recording = false;
            _created_eagerly = tf.context.executing_eagerly();
            // Enters a context inside which operations are recorded on this tape.
            if (_created_eagerly)
            {
                tf.context.ensure_initialized();
                tf.context.start_step();
            }
            _push_tape();
        }

        /// <summary>
        /// Pushes a new tape onto the tape stack.
        /// </summary>
        private void _push_tape()
        {
            if (_recording)
                throw new ValueError("Tape is still recording, This can happen if you try to " + 
                    "re-enter an already-active tape.");

            if (_tape == null)
                _tape = new Tape(_persistent, _watch_accessed_variables);
            else
                tf.GetTapeSet().Add(_tape);
           
            _recording = true;
        }

        private void _pop_tape()
        {
            if (!_recording)
                throw new ValueError("Tape is not recording.");
            _tape.PopTape(_tape);
            _recording = false;
        }

        /// <summary>
        /// Marks this tensor to be watched by the given tape.
        /// </summary>
        /// <param name="x"></param>
        public void watch(Tensor x)
        {
            _tape.Watch(x.Id);
        }

        /// <summary>
        /// Computes the gradient using operations recorded in context of this tape.
        /// </summary>
        /// <param name="target"></param>
        /// <param name="source"></param>
        /// <returns></returns>
        public Tensor gradient(Tensor target, Tensor source)
        {
            if (_recording)
            {
                if (!_persistent)
                    _pop_tape();
            }

            var results = tf.Runner.TFE_TapeGradient(_tape,
                new[] { target },
                new[] { source },
                null);

            return results[0];
        }

        public Tensor gradient(Tensor target, ResourceVariable source)
        {
            var results = gradient(target, new[] { source });

            return results[0];
        }

        public (Tensor, Tensor) gradient(Tensor target, (ResourceVariable, ResourceVariable) sources)
        {
            var results = gradient(target, new[] { sources.Item1, sources.Item2 });

            return (results[0], results[1]);
        }

        public Tensor[] gradient(Tensor target, ResourceVariable[] sources)
        {
            if (_recording)
            {
                if (!_persistent)
                    _pop_tape();
            }

            var results = tf.Runner.TFE_TapeGradient(_tape, 
                new[] { target }, 
                sources.Select(x => x.Handle).ToArray(), 
                null);

            if (!_persistent)
            {
                // Keep track of watched variables before setting tape to None
                _watched_variables = _tape.WatchedVariables();
                _tape = null;
            }

            return results;
        }

        /// <summary>
        /// Temporarily stops recording operations on this tape.
        /// </summary>
        public void stop_recording()
        {
            _pop_tape();
        }

        public void Dispose()
        {
            if (_recording)
                _pop_tape();

            if (_created_eagerly)
                tf.context.end_step();
        }
    }
}
