﻿/*****************************************************************************
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
using System.IO;
using Tensorflow.Util;

namespace Tensorflow
{
    public class op_def_registry
    {
        static Dictionary<string, OpDef> _registered_ops;

        public static Dictionary<string, OpDef> get_registered_ops()
        {
            if(_registered_ops == null)
            {
                _registered_ops = new Dictionary<string, OpDef>();
                using var buffer = new Buffer(c_api.TF_GetAllOpList());
                using var stream = buffer.DangerousMemoryBlock.Stream();
                var op_list = OpList.Parser.ParseFrom(stream);
                foreach (var op_def in op_list.Op)
                    _registered_ops[op_def.Name] = op_def;
            }

            return _registered_ops;
        }

        public static OpDef GetOpDef(string type)
        {
            var ops = get_registered_ops();
            return ops[type];
        }
    }
}
