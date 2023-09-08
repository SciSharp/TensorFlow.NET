using System.Threading;
using Tensorflow.Common.Types;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        /// <summary>
        /// Wraps `call`, applying pre- and post-processing steps.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="states"></param>
        /// <param name="training"></param>
        /// <returns></returns>
        public virtual Tensors Apply(Tensors inputs, Tensors states = null, bool? training = false, IOptionalArgs? optional_args = null)
        {
            if (callContext.Value == null)
                callContext.Value = new CallContext();

            if (_in_functional_construction_mode(inputs))
                return FunctionalConstructionCall(inputs);

            var eager = tf.executing_eagerly();
            using var ctxManager = CallContext.enter(build_graph: false);

            string nameScope = eager ? name : _name_scope();
            var scope = ops.name_scope(nameScope);
            scope.__enter__();

            if (!built)
                MaybeBuild(inputs);

            var outputs = Call(inputs, state: states, training: training);

            // memory leak
            // _set_connectivity_metadata_(inputs, outputs);
            _handle_activity_regularization(inputs, outputs);
            _set_mask_metadata(inputs, outputs, null);

            // TODO(Rinne): set save spec if null

            scope.__exit__();

            return outputs;
        }

        // TODO(Rinne): remove it and completely fix issue 1084
        [Obsolete]
        private bool _enforce_layer_construction = false;
        [Obsolete]
        internal void enforce_layer_construction()
        {
            _enforce_layer_construction = true;
        }
        [Obsolete]
        internal void unset_layer_construction()
        {
            _enforce_layer_construction = false;
        }
    }
}
