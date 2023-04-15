/*****************************************************************************
   Copyright 2020 The TensorFlow.NET Authors. All Rights Reserved.

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

using System.Collections.Generic;
using Tensorflow.Gradients;

namespace Tensorflow
{
    public partial class tensorflow
    {
        GradientTape _tapeSet;

        /// <summary>
        /// Record operations for automatic differentiation.
        /// </summary>
        /// <param name="persistent"></param>
        /// <param name="watch_accessed_variables"></param>
        /// <returns>Tape set</returns>
        public GradientTape GradientTape(bool persistent = false,
            bool watch_accessed_variables = true)
        {
            var tape = _tapeSet.PushTape(persistent: persistent,
                watch_accessed_variables: watch_accessed_variables);
            tape.StartRecord();
            return _tapeSet;
        }

        public Stack<ITape> GetTapeSet()
            => _tapeSet.GetTapeSet();

        public Tensor[] gradients(Tensor[] ys,
            Tensor[] xs,
            Tensor[] grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int? aggregation_method = null,
            Tensor[] stop_gradients = null)
        {
            return gradients_util._GradientsHelper(ys,
                xs,
                grad_ys,
                name,
                colocate_gradients_with_ops,
                gate_gradients,
                stop_gradients: stop_gradients);
        }

        public Tensor[] gradients(Tensor ys,
            Tensor[] xs,
            Tensor[] grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int? aggregation_method = null,
            Tensor[] stop_gradients = null)
        {
            return gradients_util._GradientsHelper(new Tensor[] { ys },
                xs,
                grad_ys,
                name,
                colocate_gradients_with_ops,
                gate_gradients,
                stop_gradients: stop_gradients);
        }

        public Tensor[] gradients(Tensor ys,
            Tensor xs,
            Tensor[] grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int? aggregation_method = null,
            Tensor[] stop_gradients = null)
        {
            return gradients_util._GradientsHelper(new Tensor[] { ys },
                new Tensor[] { xs },
                grad_ys,
                name,
                colocate_gradients_with_ops,
                gate_gradients,
                stop_gradients: stop_gradients);
        }
    }
}
