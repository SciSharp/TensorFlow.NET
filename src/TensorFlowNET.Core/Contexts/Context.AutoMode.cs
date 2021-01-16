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

namespace Tensorflow.Contexts
{
    /// <summary>
    /// Environment in which eager operations execute.
    /// </summary>
    public sealed partial class Context
    {
        public T RunInAutoMode<T>(Func<T> graphAction, Func<T> eagerAction, params object[] args)
        {
            if (tf.Context.has_graph_arg(args))
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
                    return graphAction();
                }
            }
            else
            {
                if (tf.Context.executing_eagerly())
                {
                    return eagerAction();
                }
                else
                {
                    return graphAction();
                }
            }
        }

        // [DebuggerStepThrough]
        public Tensors RunInAutoMode2(Func<Tensors> graphAction,
            Func<Tensors> eagerAction,
            Action<Operation> recordGradient,
            Tensors tensors)
        {
            if (tf.Context.has_graph_arg(tensors))
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
                        recordGradient(result[0].op);
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
