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

using System.Collections.Generic;
using System.Linq;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// Specifies the ndim, dtype and shape of every input to a layer.
    /// </summary>
    public class InputSpec
    {
        public int? ndim;
        public int? min_ndim;
        Dictionary<int, int> axes;
        TensorShape shape;
        public int[] AllAxisDim;

        public InputSpec(TF_DataType dtype = TF_DataType.DtInvalid,
            int? ndim = null,
            int? min_ndim = null,
            Dictionary<int, int> axes = null,
            TensorShape shape = null)
        {
            this.ndim = ndim;
            if (axes == null)
                axes = new Dictionary<int, int>();
            this.axes = axes;
            this.min_ndim = min_ndim;
            this.shape = shape;
            if (ndim == null && shape != null)
                this.ndim = shape.ndim;

            if (axes != null)
                AllAxisDim = axes.Select(x => x.Value).ToArray();
        }

        public override string ToString()
            => $"ndim={ndim}, min_ndim={min_ndim}, axes={axes.Count}";
    }
}
