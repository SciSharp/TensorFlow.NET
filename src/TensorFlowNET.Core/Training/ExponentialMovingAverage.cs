using System;
using System.Collections.Generic;
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
        Dictionary<IVariableV1, IVariableV1> _averages;

        public ExponentialMovingAverage(float decay, int? num_updates = null, bool zero_debias = false,
            string name = "ExponentialMovingAverage")
        {
            _decay = decay;
            _num_updates = num_updates;
            _zero_debias = zero_debias;
            _name = name;
            _averages = new Dictionary<IVariableV1, IVariableV1>();
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

            foreach (var var in var_list)
            {
                if (!_averages.ContainsKey(var))
                {
                    ops.init_scope();
                    var slot_creator = new SlotCreator();
                    var value = var.initialized_value();
                    var avg = slot_creator.create_slot(var,
                        value,
                        name,
                        colocate_with_primary: true);
                    ops.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var);
                    _averages[var] = avg;
                }
                else
                {
                    // avg = slot_creator.create_zeros_slot(
                    throw new NotImplementedException("");
                }
            }

            return tf_with(ops.name_scope(name), scope =>
            {
                var decay = ops.convert_to_tensor(_decay, name: "decay");
                if (_num_updates.HasValue)
                {
                    throw new NotImplementedException("ExponentialMovingAverage.apply");
                }

                var updates = new List<Tensor>();
                foreach (var var in var_list)
                {
                    var zero_debias = false;// _averages[var] in zero_debias_true
                    var ama = moving_averages.assign_moving_average(_averages[var], var, decay, zero_debias: zero_debias);
                    updates.Add(ama);
                }

                return control_flow_ops.group(updates.ToArray(), name: scope);
            });
        }
    }
}
