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

namespace Tensorflow.Framework
{
    public static class common_shapes
    {
        /// <summary>
        /// Returns the broadcasted shape between `shape_x` and `shape_y
        /// </summary>
        /// <param name="shape_x"></param>
        /// <param name="shape_y"></param>
        public static Tensor broadcast_shape(Tensor shape_x, Tensor shape_y)
        {
            var return_dims = _broadcast_shape_helper(shape_x, shape_y);
            // return tensor_shape(return_dims);
            throw new NotFiniteNumberException();
        }
        /// <summary>
        /// Helper functions for is_broadcast_compatible and broadcast_shape.
        /// </summary>
        /// <param name="shape_x"> A `Shape`</param>
        /// <param name="shape_y"> A `Shape`</param>
        /// <return> Returns None if the shapes are not broadcast compatible,
        /// a list of the broadcast dimensions otherwise.
        /// </return>
        public static Tensor _broadcast_shape_helper(Tensor shape_x, Tensor shape_y)
        {
            throw new NotFiniteNumberException();
        }

        public static int? rank(Tensor tensor)
        {
            return tensor.rank;
        }

        public static bool has_fully_defined_shape(Tensor tensor)
        {
            return tensor.shape.IsFullyDefined;
        }
    }
}
