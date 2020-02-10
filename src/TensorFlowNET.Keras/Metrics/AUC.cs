using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class AUC : Metric
    {
        public AUC(int num_thresholds= 200, string curve= "ROC", string summation_method= "interpolation",
                    string name= null, string dtype= null, float thresholds= 0.5f,
                    bool multi_label= false, Tensor label_weights= null) : base(name, dtype)
        {
            throw new NotImplementedException();
        }

        private void _build(TensorShape shape) => throw new NotImplementedException();

        public Tensor interpolate_pr_auc() => throw new NotImplementedException();

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
