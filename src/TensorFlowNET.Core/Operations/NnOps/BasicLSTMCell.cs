using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorflow.Binding;
using Tensorflow.Operations.Activation;
using Tensorflow.Keras.Engine;
using Tensorflow.Operations;

namespace Tensorflow
{
    /// <summary>
    /// Basic LSTM recurrent network cell.
    /// The implementation is based on: http://arxiv.org/abs/1409.2329.
    /// </summary>
    public class BasicLSTMCell : LayerRnnCell
    {
        int _num_units;
        float _forget_bias;
        bool _state_is_tuple;
        IActivation _activation;

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
        public BasicLSTMCell(int num_units, float forget_bias = 1.0f, bool state_is_tuple = true,
            IActivation activation = null, bool? reuse = null, string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid) : base(_reuse: reuse, name: name, dtype: dtype)
        {
            input_spec = new InputSpec(ndim: 2);
            _num_units = num_units;
            _forget_bias = forget_bias;
            _state_is_tuple = state_is_tuple;
            _activation = activation;
            if (_activation == null)
                _activation = tf.nn.tanh();
        }

        public LSTMStateTuple state_size
        {
            get
            {
                return _state_is_tuple ? 
                    new LSTMStateTuple(_num_units, _num_units) : 
                    (LSTMStateTuple)(2 * _num_units);
            }
        }
    }
}
