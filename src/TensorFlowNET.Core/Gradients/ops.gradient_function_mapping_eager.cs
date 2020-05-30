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
using System.Reflection;
using Tensorflow.Eager;
using Tensorflow.Gradients;

namespace Tensorflow
{
    public partial class ops
    {
        public static Dictionary<string, Func<EagerOperation, IntPtr[], EagerTensor[]>> gradientFunctionsEager = null;

        public static void RegisterFromAssemblyEager()
        {
            if (gradientFunctionsEager == null)
            {
                gradientFunctionsEager = new Dictionary<string, Func<EagerOperation, IntPtr[], EagerTensor[]>>();

                var gradGroups = Assembly.GetExecutingAssembly()
                    .GetTypes()
                    .Where(x => x.GetCustomAttribute<RegisterGradientEager>() != null)
                    .ToArray();

                foreach (var g in gradGroups)
                {
                    var methods = g.GetMethods()
                        .Where(x => x.GetCustomAttribute<RegisterGradientEager>() != null)
                        .ToArray();

                    foreach (var m in methods)
                    {
                        RegisterGradientFunctionEager(m.GetCustomAttribute<RegisterGradientEager>().Name,
                            (oper, out_grads) =>
                                 g.InvokeMember(m.Name,
                                    BindingFlags.InvokeMethod,
                                    null,
                                    null,
                                    args: new object[] { oper, out_grads }) as EagerTensor[]
                        );
                    }

                    // REGISTER_NO_GRADIENT_OP
                    methods = g.GetMethods()
                        .Where(x => x.GetCustomAttribute<RegisterNoGradient>() != null)
                        .ToArray();

                    foreach (var m in methods)
                        RegisterNoGradientFunctionEager(m.GetCustomAttribute<RegisterNoGradient>().Name);
                }
            }
        }

        /// <summary>
        /// Regiter new gradient function
        /// </summary>
        /// <param name="name">operation type</param>
        /// <param name="func">function delegate</param>
        public static void RegisterGradientFunctionEager(string name, Func<EagerOperation, IntPtr[], EagerTensor[]> func)
        {
            RegisterFromAssemblyEager();

            gradientFunctionsEager[name] = func;
        }

        public static void RegisterNoGradientFunctionEager(string name)
        {
            RegisterFromAssemblyEager();

            gradientFunctionsEager[name] = null;
        }

        public static Func<EagerOperation, IntPtr[], EagerTensor[]> get_gradient_function_eager(EagerOperation op)
        {
            if (op.inputs == null) return null;

            RegisterFromAssemblyEager();

            if (!gradientFunctionsEager.ContainsKey(op.type))
                throw new LookupError($"can't get graident function through get_gradient_function {op.type}");

            return gradientFunctionsEager[op.type];
        }
    }
}
