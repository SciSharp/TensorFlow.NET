using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        /// <summary>
        /// Wraps `call`, applying pre- and post-processing steps.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="state"></param>
        /// <param name="is_training"></param>
        /// <returns></returns>
        public Tensors Apply(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            callContext = callContext ?? new ThreadLocal<CallContext>()
            {
                Value = new CallContext()
            };

            if (_in_functional_construction_mode(inputs))
                return FunctionalConstructionCall(inputs);

            Tensors outputs = null;

            var eager = tf.executing_eagerly();
            using var ctxManager = CallContext.enter();

            string nameScope = "";
            if (eager)
                nameScope = Name;
            else
                nameScope = _name_scope();

            if (!inputs.IsEagerTensor)
                tf.Context.graph_mode();

            tf_with(ops.name_scope(nameScope), scope =>
            {
                if (!built)
                    MaybeBuild(inputs);

                outputs = call_fn(inputs, state: state, is_training: is_training);

                outputs = _set_connectivity_metadata_(inputs, outputs);
                _handle_activity_regularization(inputs, outputs);
                _set_mask_metadata(inputs, outputs, null);
            });

            if (!inputs.IsEagerTensor)
                tf.Context.restore_mode();

            return outputs;
        }
    }
}
