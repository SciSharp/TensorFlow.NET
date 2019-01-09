using System;
using System.Collections.Generic;
using System.Text;
using distribute_lib = Tensorflow.Distribute;

namespace Tensorflow
{
    /// <summary>
    /// Base class for optimizers.
    /// This class defines the API to add Ops to train a model.  You never use this
    /// class directly, but instead instantiate one of its subclasses such as
    /// `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.
    /// </summary>
    public abstract class Optimizer
    {
        public string Name { get; set; }
        public double LearningRate { get; set; }
        public Tensor LearningRateTensor { get; set; }

        public Optimizer(double learning_rate, bool use_locking, string name = "")
        {
            if (String.IsNullOrEmpty(name))
                throw new NotImplementedException("Must specify the optimizer name");

            Name = name;
        }

        /// <summary>
        /// Add operations to minimize `loss` by updating `var_list`
        /// </summary>
        /// <param name="loss"></param>
        /// <returns></returns>
        public Optimizer minimize(Tensor loss, GateGradientType gate_gradients = GateGradientType.GATE_OP)
        {
            compute_gradients(loss, gate_gradients);
            return this;
        }

        /// <summary>
        /// Compute gradients of `loss` for the variables in `var_list`.
        /// </summary>
        /// <param name="loss"></param>
        /// <param name="gate_gradients"></param>
        public List<KeyValuePair<object, object>> compute_gradients(Tensor loss, GateGradientType gate_gradients = GateGradientType.GATE_OP)
        {
            int num_towers = 1;
            if(distribute_lib.get_loss_reduction() == VariableAggregationType.MEAN)
            {
                
            }

            var var_list = variables.trainable_variables();
            return null;
        }
    }
}
