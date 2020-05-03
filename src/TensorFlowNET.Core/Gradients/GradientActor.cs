using System;
using System.Collections.Generic;
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
    public class GradientActor : IDisposable
    {
        bool _recording;
        bool _persistent;
        bool _watch_accessed_variables;
        bool _created_eagerly;
        Tape _tape;

        public GradientActor(bool persistent = false,
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

        public Tensor gradient(Tensor target, Tensor sources)
        {
            if(_recording)
            {
                if (!_persistent)
                    _pop_tape();
            }

            using var status = new Status();
            var et = c_api.TFE_TapeGradient(_tape, 
                new IntPtr[] { (target as EagerTensor).EagerTensorHandle }, 1,
                new IntPtr[] { (sources as EagerTensor).EagerTensorHandle }, 1,
                status);
            status.Check(true);
            return et;
        }

        public void Dispose()
        {
            
        }
    }
}
