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
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// Specifies the ndim, dtype and shape of every input to a layer.
    /// </summary>
    public class InputSpec: IKerasConfigable
    {
        public int? ndim;
        public int? max_ndim;
        public int? min_ndim;
        Dictionary<int, int> axes;
        Shape shape;
        TF_DataType dtype;
        public int[] AllAxisDim;

        public InputSpec(TF_DataType dtype = TF_DataType.DtInvalid,
            int? ndim = null,
            int? min_ndim = null,
            int? max_ndim = null,
            Dictionary<int, int> axes = null,
            Shape shape = null)
        {
            this.ndim = ndim;
            if (axes == null)
                axes = new Dictionary<int, int>();
            this.axes = axes;
            this.min_ndim = min_ndim;
            this.max_ndim = max_ndim;
            this.shape = shape;
            this.dtype = dtype;
            if (ndim == null && shape != null)
                this.ndim = shape.ndim;

            if (axes != null)
                AllAxisDim = axes.Select(x => x.Value).ToArray();
        }

       public IKerasConfig get_config()
        {
            return new Config()
            {
                DType = dtype == TF_DataType.DtInvalid ? null : dtype,
                Shape = shape,
                Ndim = ndim,
                MinNdim = min_ndim,
                MaxNdim = max_ndim,
                Axes = axes.ToDictionary(x => x.Key.ToString(), x => x.Value)
            };
        }

        public override string ToString()
            => $"ndim={ndim}, min_ndim={min_ndim}, axes={axes.Count}";

        public class Config: IKerasConfig
        {
            public TF_DataType? DType { get; set; }
            public Shape Shape { get; set; }
            public int? Ndim { get; set; }
            public int? MinNdim { get;set; }
            public int? MaxNdim { get;set; }
            public IDictionary<string, int> Axes { get; set; }
        }
    }
}
