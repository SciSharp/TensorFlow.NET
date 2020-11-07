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
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Contexts;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        public (TF_DataType, Tensor[]) ArgsToMatchingEager(Context ctx, TF_DataType default_dtype = TF_DataType.DtInvalid, object[] args = null)
        {
            if (args.Length == 0 && default_dtype != TF_DataType.DtInvalid)
                return (default_dtype, null);

            if (args.Count(x => x is Tensor) == args.Length)
                return ((args[0] as Tensor).dtype, args.Select(x => x as Tensor).ToArray());

            var dtype = TF_DataType.DtInvalid;
            foreach (var x in args)
            {
                if (x is Tensor et)
                    dtype = et.dtype;
            }

            if (dtype == TF_DataType.DtInvalid)
            {
                var ret = new List<Tensor>();
                foreach (var t in args)
                {
                    ret.Add(ops.convert_to_tensor(t, dtype, preferred_dtype: default_dtype, ctx: ctx) as Tensor);
                    if (dtype == TF_DataType.DtInvalid)
                        dtype = ret.Last().dtype;
                }

                return (dtype, ret.ToArray());
            }
            else
                throw new NotImplementedException("");
        }
    }
}
