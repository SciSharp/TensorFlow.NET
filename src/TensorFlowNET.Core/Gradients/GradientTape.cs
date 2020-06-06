using Google.Protobuf.WellKnownTypes;
using NumSharp.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
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
        static bool _recording;
        public static bool Recording => _recording;
        bool _persistent;
        bool _watch_accessed_variables;
        ResourceVariable[] _watched_variables;
        bool _created_eagerly;
        Tape _tape;

        public GradientTape(bool persistent = false,
            bool watch_accessed_variables = true)
        {
            _persistent = persistent;
            _watch_accessed_variables = watch_accessed_variables;
            _created_eagerly = tf.context.executing_eagerly();
            _push_tape();
        }

        private void _push_tape()
        {
            if (_recording)
                throw new ValueError("Tape is still recording, This can happen if you try to " + 
                    "re-enter an already-active tape.");

            if (_tape == null)
                _tape = new Tape(_persistent, _watch_accessed_variables);
            else
                throw new NotImplementedException("");

            _recording = true;
        }

        private void _pop_tape()
        {
            if (!_recording)
                throw new ValueError("Tape is not recording.");
            _tape.pop_tape(_tape);
            _recording = false;
        }

        /// <summary>
        /// Marks this tensor to be watched by the given tape.
        /// </summary>
        /// <param name="x"></param>
        public void watch(Tensor x)
        {
            _tape.watch(x as EagerTensor);
        }

        public Tensor gradient(Tensor target, Tensor source)
        {
            if (_recording)
            {
                if (!_persistent)
                    _pop_tape();
            }

            var results = new[] { new EagerTensor() };
            Status status = c_api.TFE_TapeGradient(_tape,
                new[] { (target as EagerTensor).EagerTensorHandle }, 1,
                new[] { (source as EagerTensor).EagerTensorHandle }, 1,
                results.Select(x => x.EagerTensorHandle).ToArray(), results.Length);
            status.Check(true);
            return results[0].Resolve();
        }

        public (Tensor, Tensor) gradient(Tensor target, (ResourceVariable, ResourceVariable) sources)
        {
            var results = gradient(target as EagerTensor, new[] { sources.Item1, sources.Item2 });

            return (results[0], results[1]);
        }

        public EagerTensor[] gradient(EagerTensor target, ResourceVariable[] sources)
        {
            if (_recording)
            {
                if (!_persistent)
                    _pop_tape();
            }

            var results = sources.Select(x => new EagerTensor()).ToArray();
            Status status = c_api.TFE_TapeGradient(_tape,
                new IntPtr[]
                {
                    target.EagerTensorHandle
                }, 1,
                sources.Select(x => (x.Handle as EagerTensor).EagerTensorHandle).ToArray(), sources.Length,
                results.Select(x => x.EagerTensorHandle).ToArray(), results.Length);
            status.Check(true);

            if (!_persistent)
            {
                // Keep track of watched variables before setting tape to None
                _watched_variables = _tape.watched_variables();
                _tape = null;
            }

            return results.Select(x => x.Resolve()).ToArray();
        }

        public void Dispose()
        {
            if (_recording)
                _pop_tape();
        }
    }
}
