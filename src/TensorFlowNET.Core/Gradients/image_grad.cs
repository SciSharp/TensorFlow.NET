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

using System;
using System.Linq;

namespace Tensorflow.Gradients
{
    [RegisterGradient("image_grad")]
    public class image_grad
    {
        [RegisterGradient("ResizeNearestNeighbor")]
        public static Tensor[] _ResizeNearestNeighborGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var image = op.inputs[0];
            var shape = new TensorShape(image.shape.Skip(1).Take(2).ToArray());
            Tensor image_shape = null;
            if (shape.is_fully_defined())
                image_shape = constant_op.constant(image.shape[1..3]);
            else
                image_shape = array_ops.shape(image)["1:3"];

            grad = gen_image_ops.resize_nearest_neighbor_grad(
              grad,
              image_shape,
              align_corners: op.get_attr<bool>("align_corners"),
              half_pixel_centers: op.get_attr<bool>("half_pixel_centers"));

            return new Tensor[]
            {
                grad,
                null
            };
        }
    }
}
