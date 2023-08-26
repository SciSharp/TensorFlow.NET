using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras
{
    public interface IOptimizerApi
    {
        /// <summary>
        /// Adam optimization is a stochastic gradient descent method that is based on
        /// adaptive estimation of first-order and second-order moments.
        /// </summary>
        /// <param name="learning_rate"></param>
        /// <param name="beta_1"></param>
        /// <param name="beta_2"></param>
        /// <param name="epsilon"></param>
        /// <param name="amsgrad"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        IOptimizer Adam(float learning_rate = 0.001f,
                float beta_1 = 0.9f,
                float beta_2 = 0.999f,
                float epsilon = 1e-7f,
                bool amsgrad = false,
                string name = "Adam");

        /// <summary>
        /// Adam enables L2 weight decay on gradients.
        /// </summary>
        /// <param name="learning_rate"></param>
        /// <param name="weight_decay"></param>
        /// <param name="beta_1"></param>
        /// <param name="beta_2"></param>
        /// <param name="epsilon"></param>
        /// <param name="amsgrad"></param>
        /// <param name="decay_params"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        IOptimizer AdamW(float learning_rate = 0.001f,
                float weight_decay = 0.004f,
                float beta_1 = 0.9f,
                float beta_2 = 0.999f,
                float epsilon = 1e-7f,
                bool amsgrad = false,
                List<string> no_decay_params = null,
                string name = "AdamW");

        /// <summary>
        /// Construct a new RMSprop optimizer.
        /// </summary>
        /// <param name="learning_rate"></param>
        /// <param name="rho"></param>
        /// <param name="momentum"></param>
        /// <param name="epsilon"></param>
        /// <param name="centered"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        IOptimizer RMSprop(float learning_rate = 0.001f,
                float rho = 0.9f,
                float momentum = 0.0f,
                float epsilon = 1e-7f,
                bool centered = false,
                string name = "RMSprop");

        IOptimizer SGD(float learning_rate = 0.01f, float momentum = 0f);
    }
}
