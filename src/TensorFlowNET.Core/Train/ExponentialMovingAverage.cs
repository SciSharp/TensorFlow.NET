using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Train
{
    public class ExponentialMovingAverage
    {
        float _decay;
        int? _num_updates;
        bool _zero_debias;
        string _name;
        public string name => _name;

        public ExponentialMovingAverage(float decay, int? num_updates = null, bool zero_debias = false,
            string name = "ExponentialMovingAverage")
        {
            _decay = decay;
            _num_updates = num_updates;
            _zero_debias = zero_debias;
            _name = name;
        }

        /// <summary>
        /// Maintains moving averages of variables.
        /// </summary>
        /// <param name="var_list"></param>
        /// <returns></returns>
        public Operation apply(VariableV1[] var_list = null)
        {
            throw new NotImplementedException("");
        }


    }
}
