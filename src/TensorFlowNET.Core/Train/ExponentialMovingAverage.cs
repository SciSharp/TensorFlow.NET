using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Train
{
    public class ExponentialMovingAverage
    {
        float _decay;
        int? _num_updates;
        bool _zero_debias;
        string _name;
        public string name => _name;
        List<VariableV1> _averages;

        public ExponentialMovingAverage(float decay, int? num_updates = null, bool zero_debias = false,
            string name = "ExponentialMovingAverage")
        {
            _decay = decay;
            _num_updates = num_updates;
            _zero_debias = zero_debias;
            _name = name;
            _averages = new List<VariableV1>();
        }

        /// <summary>
        /// Maintains moving averages of variables.
        /// </summary>
        /// <param name="var_list"></param>
        /// <returns></returns>
        public Operation apply(RefVariable[] var_list = null)
        {
            if (var_list == null)
                var_list = variables.trainable_variables() as RefVariable[];

            foreach(var var in var_list)
            {
                if (!_averages.Contains(var))
                {
                    ops.init_scope();
                    var slot = new SlotCreator();
                    var.initialized_value();
                    // var avg = slot.create_zeros_slot
                }
            }

            throw new NotImplementedException("");
        }
    }
}
