using MethodBoundaryAspect.Fody.Attributes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    [DebuggerStepThrough]
    public sealed class AutoNumPyAttribute : OnMethodBoundaryAspect
    {
        bool _changedMode = false;
        bool _locked = false;
        static object locker = new Object();
        public override void OnEntry(MethodExecutionArgs args)
        {
            Monitor.Enter(locker, ref _locked);
            
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

            if (_locked)
                Monitor.Exit(locker);
        }
    }
}
