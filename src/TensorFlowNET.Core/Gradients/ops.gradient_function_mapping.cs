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

                                var results = m.Name switch
                                {
                                    /*"_AddGrad" => math_grad._AddGrad(oper, out_grads),
                                    "_AddV2Grad" => math_grad._AddV2Grad(oper, out_grads),
                                    "_BiasAddGrad" => nn_grad._BiasAddGrad(oper, out_grads),
                                    "_CastGrad" => math_grad._CastGrad(oper, out_grads),
                                    "_ConcatGradV2" => array_grad._ConcatGradV2(oper, out_grads),
                                    "_Conv2DGrad" => nn_grad._Conv2DGrad(oper, out_grads),
                                    "_ExpandDimsGrad" => array_grad._ExpandDimsGrad(oper, out_grads),
                                    "_ExpGrad" => math_grad._ExpGrad(oper, out_grads),
                                    "_FusedBatchNormV3Grad" => nn_grad._FusedBatchNormV3Grad(oper, out_grads),
                                    "_IdGrad" => math_grad._IdGrad(oper, out_grads),
                                    "_LeakyReluGrad" => nn_grad._LeakyReluGrad(oper, out_grads),
                                    "_Log1pGrad" => math_grad._Log1pGrad(oper, out_grads),
                                    "_MaximumGrad" => math_grad._MaximumGrad(oper, out_grads),
                                    "_MeanGrad" => math_grad._MeanGrad(oper, out_grads),
                                    "_MinimumGrad" => math_grad._MinimumGrad(oper, out_grads),
                                    "_MulGrad" => math_grad._MulGrad(oper, out_grads),
                                    "_NegGrad" => math_grad._NegGrad(oper, out_grads),
                                    "_PadGrad" => array_grad._PadGrad(oper, out_grads),
                                    "_PowGrad" => math_grad._PowGrad(oper, out_grads),
                                    "_RealDivGrad" => math_grad._RealDivGrad(oper, out_grads),
                                    "_ReadGrad" => resource_variable_grad._ReadGrad(oper, out_grads),
                                    "_ReshapeGrad" => array_grad._ReshapeGrad(oper, out_grads),
                                    "_ResizeNearestNeighborGrad" => image_grad._ResizeNearestNeighborGrad(oper, out_grads),
                                    "_SelectGrad" => math_grad._SelectGrad(oper, out_grads),
                                    "_SigmoidGrad" => math_grad._SigmoidGrad(oper, out_grads),
                                    "_SumGrad" => math_grad._SumGrad(oper, out_grads),
                                    "_SubGrad" => math_grad._SubGrad(oper, out_grads),
                                    "_StridedSliceGrad" => array_grad._StridedSliceGrad(oper, out_grads),*/
                                    _ => g.InvokeMember(m.Name,
                                           BindingFlags.InvokeMethod,
                                           null,
                                           null,
                                           args: new object[] { oper, out_grads }) as Tensor[]
                                };

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

            RegisterFromAssembly();

            if (!gradientFunctions.ContainsKey(op.type))
                throw new LookupError($"can't get graident function through get_gradient_function {op.type}");

            return gradientFunctions[op.type];
        }
    }
}
