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
using Tensorflow.Gradients;

namespace Tensorflow
{
    public partial class ops
    {
        public static Dictionary<string, Func<Operation, Tensor[], Tensor[]>> gradientFunctions = null;

        public static void RegisterFromAssembly()
        {
            if (gradientFunctions == null)
            {
                gradientFunctions = new Dictionary<string, Func<Operation, Tensor[], Tensor[]>>();

                var gradGroups = Assembly.GetExecutingAssembly()
                    .GetTypes()
                    .Where(x => x.GetCustomAttribute<RegisterGradient>() != null)
                    .ToArray();

                foreach (var g in gradGroups)
                {
                    var methods = g.GetMethods()
                        .Where(x => x.GetCustomAttribute<RegisterGradient>() != null)
                        .ToArray();

                    foreach (var m in methods)
                    {
                        RegisterGradientFunction(m.GetCustomAttribute<RegisterGradient>().Name,
                            (oper, out_grads) =>
                            {
                                // tf.Logger.Debug($"Caculate Gradient: {oper.name} {m.Name}");

                                var results = g.InvokeMember(m.Name,
                                    BindingFlags.InvokeMethod,
                                    null,
                                    null,
                                    args: new object[] { oper, out_grads }) as Tensor[];

                                // foreach (var result in results.Where(x => x != null))
                                    // tf.Logger.Debug($"Gradient: {result.name} {result.shape}");

                                return results;
                            }
                        );
                    }

                    // REGISTER_NO_GRADIENT_OP
                    methods = g.GetMethods()
                        .Where(x => x.GetCustomAttribute<RegisterNoGradient>() != null)
                        .ToArray();

                    foreach (var m in methods)
                        RegisterNoGradientFunction(m.GetCustomAttribute<RegisterNoGradient>().Name);
                }
            }
        }

        /// <summary>
        /// Regiter new gradient function
        /// </summary>
        /// <param name="name">operation type</param>
        /// <param name="func">function delegate</param>
        public static void RegisterGradientFunction(string name, Func<Operation, Tensor[], Tensor[]> func)
        {
            RegisterFromAssembly();

            gradientFunctions[name] = func;
        }

        public static void RegisterNoGradientFunction(string name)
        {
            RegisterFromAssembly();

            gradientFunctions[name] = null;
        }

        public static Func<Operation, Tensor[], Tensor[]> get_gradient_function(Operation op)
        {
            if (op.inputs == null) return null;

            var gradient_function = op._gradient_function;
            if(gradient_function is null)
            {
                RegisterFromAssembly();

                if (!gradientFunctions.ContainsKey(op.type))
                    throw new LookupError($"can't get graident function through get_gradient_function {op.type}");

                return gradientFunctions[op.type];
            }

            Tensor[] wrapped_gradient_function(Operation operation, Tensor[] args)
            {
                return gradient_function(operation, args);
            }
            // TODO(Rinne): check if this needs to be registered.
            return wrapped_gradient_function;
        }
    }
}
