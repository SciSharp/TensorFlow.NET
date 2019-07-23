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

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor[] gradients(Tensor[] ys,
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

        public static Tensor[] gradients(Tensor ys,
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

        public static Tensor[] gradients(Tensor ys,
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
