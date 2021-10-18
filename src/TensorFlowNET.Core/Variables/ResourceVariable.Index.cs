/*****************************************************************************
   Copyright 2020 Haiping Chen. All Rights Reserved.

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

using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using System.Linq;

namespace Tensorflow
{
    public partial class ResourceVariable
    {
        public Tensor this[params Slice[] slices]
        {
            get
            {
                var args = tensor_util.ParseSlices(slices);

                return tf_with(ops.name_scope(null, "strided_slice", args), scope =>
                {
                    string name = scope;
                    if (args.Begin != null)
                    {
                        (args.PackedBegin, args.PackedEnd, args.PackedStrides) =
                            (array_ops.stack(args.Begin),
                                array_ops.stack(args.End),
                                array_ops.stack(args.Strides));

                        var tensor = gen_array_ops.strided_slice(
                            this,
                            args.PackedBegin,
                            args.PackedEnd,
                            args.PackedStrides,
                            begin_mask: args.BeginMask,
                            end_mask: args.EndMask,
                            shrink_axis_mask: args.ShrinkAxisMask,
                            new_axis_mask: args.NewAxisMask,
                            ellipsis_mask: args.EllipsisMask,
                            name: name);

                        tensor.OriginalVar = this;
                        tensor.OriginalVarSlice = args;

                        return tensor;
                    }

                    throw new NotImplementedException("");
                });
            }
        }

        public Tensor this[params string[] slices]
            => this[slices.Select(x => new Slice(x)).ToArray()];
    }
}
