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
using System.Diagnostics;
using System.Linq;
using Tensorflow.Eager;
using static Tensorflow.Binding;
using Google.Protobuf;
using System.Collections.Generic;

namespace Tensorflow.Contexts
{
    /// <summary>
    /// Environment in which eager operations execute.
    /// </summary>
    public sealed partial class Context
    {
        // [DebuggerStepThrough]
        public Tensors ExecuteOp(string OpType, string Name, ExecuteOpArgs args)
        {
            Func<Tensors> graphAction = () =>
            {
                var keywords = new Dictionary<string, object>();
                if(args.OpInputArgs != null)
                {
                    foreach (var (i, input) in enumerate(args.OpInputArgs))
                        keywords[$"input_{i}"] = input;
                }

                if(args.OpAttrs != null)
                {
                    foreach (var attr in args.OpAttrs)
                        keywords[attr.Key] = attr.Value;
                }

                return tf.OpDefLib._apply_op_helper(OpType, Name, keywords).outputs;
            };

            Func<Tensors> eagerAction = () =>
            {
                var opExecInfo = new FastPathOpExecInfo(OpType, Name, args.OpInputArgs)
                {
                    attrs = args.OpAttrs
                };
                return tf.Runner.TFE_FastPathExecute(opExecInfo);
            };

            if (tf.Context.has_graph_arg(args.OpInputArgs))
            {
                if (executing_eagerly())
                {
                    graph_mode();
                    var result = graphAction();
                    restore_mode();
                    return result;
                }
                else
                {
                    var result = graphAction();
                    if (tf.Runner.MustRecordGradient())
                    {
                        var op = result[0].op;
                        Dictionary<string, object> attrs;
                        if (args.GetGradientAttrs == null)
                        {
                            attrs = new Dictionary<string, object>();
                            attrs["T"] = op.get_attr<TF_DataType>("T");
                        }
                        else
                        {
                            attrs = ConvertToDict(args.GetGradientAttrs(op));
                        }
                        var args1 = new object[attrs.Count() * 2];
                        int i = 0;
                        foreach (var arg in attrs)
                        {
                            args1[i] = arg.Key;
                            args1[i + 1] = arg.Value;
                            i += 2;
                        }
                        tf.Runner.RecordGradient(OpType, op.inputs, args1, op.outputs);
                    }
                    return result;
                }
            }
            else
            {
                return eagerAction();
            }
        }
    }
}
