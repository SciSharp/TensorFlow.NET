using System;
using System.Linq;
using Tensorflow.Keras.Engine;
using Tensorflow.Operations;
using Tensorflow.Operations.Activation;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// Basic LSTM recurrent network cell.
    /// The implementation is based on: http://arxiv.org/abs/1409.2329.
    /// </summary>
    public class BasicLstmCell : LayerRnnCell
    {
        int _num_units;
        float _forget_bias;
        bool _state_is_tuple;
        IActivation _activation;
        LSTMStateTuple _state;
        IVariableV1 _kernel;
        IVariableV1 _bias;
        string _WEIGHTS_VARIABLE_NAME = "kernel";
        string _BIAS_VARIABLE_NAME = "bias";

        /// <summary>
        /// Initialize the basic LSTM cell.
        /// </summary>
        /// <param name="num_units">The number of units in the LSTM cell.</param>
        /// <param name="forget_bias"></param>
        /// <param name="state_is_tuple"></param>
        /// <param name="activation"></param>
        /// <param name="reuse"></param>
        /// <param name="name"></param>
        /// <param name="dtype"></param>
        public BasicLstmCell(int num_units, float forget_bias = 1.0f, bool state_is_tuple = true,
            IActivation activation = null, bool? reuse = null, string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid) : base(_reuse: reuse, name: name, dtype: dtype)
        {
            inputSpec = new InputSpec(ndim: 2);
            _num_units = num_units;
            _forget_bias = forget_bias;
            _state_is_tuple = state_is_tuple;
            _activation = activation;
            if (_activation == null)
                _activation = tf.nn.tanh();
        }

        protected override void build(TensorShape input_shape)
        {
            var input_depth = input_shape.dims.Last();
            var h_depth = _num_units;
            _kernel = add_weight(_WEIGHTS_VARIABLE_NAME,
                shape: new[] { input_depth + h_depth, 4 * _num_units });
            _bias = add_weight(_BIAS_VARIABLE_NAME,
                shape: new[] { 4 * _num_units },
                initializer: tf.zeros_initializer);
            built = true;
        }

        public Tensor __call__(Tensor inputs, LSTMStateTuple state)
        {
            _state = state;
            return base.__call__(inputs);
        }

        /// <summary>
        /// Long short-term memory cell (LSTM).
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="training"></param>
        /// <param name="state"></param>
        /// <returns></returns>
        protected Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            var one = constant_op.constant(1, dtype: dtypes.int32);
            // Parameters of gates are concatenated into one multiply for efficiency.
            Tensor c = null;
            Tensor h = null;
            if (_state_is_tuple)
                (c, h) = ((Tensor)_state.c, (Tensor)_state.h);
            else
            {
                // array_ops.split(value: state, num_or_size_splits: 2, axis: one);
                throw new NotImplementedException("BasicLstmCell call");
            }
            var gate_inputs = math_ops.matmul(array_ops.concat(new[] { (Tensor)inputs, h }, 1), _kernel.AsTensor());
            gate_inputs = nn_ops.bias_add(gate_inputs, _bias);

            // i = input_gate, j = new_input, f = forget_gate, o = output_gate
            var tensors = array_ops.split(value: gate_inputs, num_split: 4, axis: one);
            var (i, j, f, o) = (tensors[0], tensors[1], tensors[2], tensors[3]);

            var forget_bias_tensor = constant_op.constant(_forget_bias, dtype: f.dtype);
            // Note that using `add` and `multiply` instead of `+` and `*` gives a
            // performance improvement. So using those at the cost of readability.
            var new_c = gen_math_ops.add(
                math_ops.multiply(c, math_ops.sigmoid(gen_math_ops.add(f, forget_bias_tensor))),
                math_ops.multiply(math_ops.sigmoid(i), _activation.Activate(j)));

            var new_h = math_ops.multiply(_activation.Activate(new_c), math_ops.sigmoid(o));


            if (_state_is_tuple)
                return new_c;
            else
                return array_ops.concat(new[] { new_c, new_h }, 1);
        }

        public override object get_initial_state(Tensor inputs = null, Tensor batch_size = null, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            if (inputs != null)
                throw new NotImplementedException("get_initial_state input is not null");

            return zero_state(batch_size, dtype);
        }

        /// <summary>
        /// Return zero-filled state tensor(s).
        /// </summary>
        /// <param name="batch_size"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        private LSTMStateTuple zero_state(Tensor batch_size, TF_DataType dtype)
        {
            LSTMStateTuple output = null;
            tf_with(ops.name_scope($"{GetType().Name}ZeroState", values: new { batch_size }), delegate
            {
                output = _zero_state_tensors(state_size, batch_size, dtype);
            });

            return output;
        }

        private LSTMStateTuple _zero_state_tensors(object state_size, Tensor batch_size, TF_DataType dtype)
        {
            if (state_size is LSTMStateTuple state_size_tuple)
            {
                var outputs = state_size_tuple.Flatten()
                    .Select(x => (int)x)
                    .Select(s =>
                    {
                        var c = rnn_cell_impl._concat(batch_size, s);
                        var size = array_ops.zeros(c, dtype: dtype);

                        var c_static = rnn_cell_impl._concat(batch_size, s, @static: true);
                        size.set_shape(c_static);

                        return size;
                    }).ToArray();

                return new LSTMStateTuple(outputs[0], outputs[1]);
            }

            throw new NotImplementedException("_zero_state_tensors");
        }

        public override object state_size
        {
            get
            {
                if (_state_is_tuple)
                    return new LSTMStateTuple(_num_units, _num_units);
                else
                    return 2 * _num_units;
            }
        }
    }
}
