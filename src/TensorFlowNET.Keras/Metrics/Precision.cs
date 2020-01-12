using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class Precision : Metric
    {
        public Precision(float? thresholds = null, int? top_k = null, int? class_id = null, string name = null, string dtype = null) : base(name, dtype)
        {
            throw new NotImplementedException();
        }

        public Precision(float[] thresholds = null, int? top_k = null, int? class_id = null, string name = null, string dtype = null) : base(name, dtype)
        {
            throw new NotImplementedException();
        }

        public override Tensor result()
        {
            throw new NotImplementedException();
        }

        public override void update_state(Args args, KwArgs kwargs)
        {
            throw new NotImplementedException();
        }

        public override void reset_states()
        {
            throw new NotImplementedException();
        }

        public override Hashtable get_config()
        {
            throw new NotImplementedException();
        }

    }
}
