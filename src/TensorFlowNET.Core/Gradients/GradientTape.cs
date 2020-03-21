using System;
using System.Collections.Generic;
using System.Text;
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
    public class GradientTape
    {
        bool _persistent;
        bool _watch_accessed_variables;

        public GradientTape(bool persistent = false,
            bool watch_accessed_variables = true)
        {
            _persistent = persistent;
            _watch_accessed_variables = watch_accessed_variables;
        }
    }
}
