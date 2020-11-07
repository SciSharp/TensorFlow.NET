/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

namespace Tensorflow.Train
{
    /// <summary>
    /// Optimizer that implements the gradient descent algorithm.
    /// </summary>
    public class GradientDescentOptimizer : Optimizer
    {
        private bool _useTensor;

        /// <summary>
        /// Construct a new gradient descent optimizer.
        /// </summary>
        /// <param name="learning_rate">A Tensor or a floating point value.  The learning
        /// rate to use.</param>
        /// <param name="use_locking">If true use locks for update operations.</param>
        /// <param name="name">Optional name prefix for the operations created when applying
        /// gradients.Defaults to "GradientDescent".</param>
        /// <remarks>
        /// When eager execution is enabled, `learning_rate` can be a callable that
        /// takes no arguments and returns the actual value to use.This can be useful
        /// for changing these values across different invocations of optimizer
        /// functions.
        /// </remarks>
        public GradientDescentOptimizer(float learning_rate, bool use_locking = false, string name = "GradientDescent")
            : base(learning_rate, use_locking, name)
        {
            _lr = learning_rate;
            _useTensor = false;
        }

        public GradientDescentOptimizer(Tensor learning_rate, bool use_locking = false, string name = "GradientDescent")
            : base(learning_rate, use_locking, name)
        {
            _lr_t = learning_rate;
            _useTensor = true;
        }

        public override void _prepare()
        {
            if (!_useTensor)
            {
                var lr = _call_if_callable(_lr);
                _lr_t = ops.convert_to_tensor(lr, name: "learning_rate");
            }

        }
    }
}
