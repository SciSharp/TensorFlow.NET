using MethodBoundaryAspect.Fody.Attributes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    [DebuggerStepThrough]
    public sealed class AutoNumPyAttribute : OnMethodBoundaryAspect
    {
        bool _changedMode = false;

        public override void OnEntry(MethodExecutionArgs args)
        {
            if (!tf.executing_eagerly())
            {
                tf.Context.eager_mode();
                _changedMode = true;
            }
        }

        public override void OnExit(MethodExecutionArgs args)
        {
            if (_changedMode)
                tf.Context.restore_mode();
        }
    }
}
