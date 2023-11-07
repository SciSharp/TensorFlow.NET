using System;
using System.Linq;
using Tensorflow.Gradients;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        public bool RecordGradient(string op_name,
            Tensor[] inputs,
            object[] attrs,
            Tensor[] results,
            BackwardFunction backwardFunction = null)
        {
            var input_ids = MakeTensorIDList(inputs);
            var input_dtypes = MakeTensorDtypeList(inputs);
            bool should_record = false;
            foreach (var tape in tf.GetTapeSet())
            {
                if (tape.ShouldRecord(input_ids, input_dtypes))
                {
                    should_record = true;
                    break;
                }
            }

            if (!should_record)
            {
                /*for (TFE_Py_ForwardAccumulator* accumulator : SafeAccumulatorSet())
                {
                    if (accumulator->accumulator->ShouldRecord(input_ids, input_dtypes))
                    {
                        should_record = true;
                        break;
                    }
                }*/
            }
            
            if (!should_record) return should_record;
            // tf.Logger.Debug($"RecordGradient: op_name={op_name}");

            /*Tensor[] op_outputs = null;
            var unused_output_indices = gradient_exclustions.OpGradientUnusedOutputIndices(op_name);
            if (unused_output_indices != null)
            {
                if (unused_output_indices.Length == 0)
                    op_outputs = new Tensor[0];
                else
                {
                    // op_outputs = CopySequenceSettingIndicesToNull(results, *unused_output_indices);
                }
            }
            else
                op_outputs = results;

            Tensor[] op_inputs = null;
            var unused_input_indices = gradient_exclustions.OpGradientUnusedInputIndices(op_name);
            if (unused_input_indices != null)
            {
                if (unused_input_indices.Length == 0)
                    op_inputs = new Tensor[0];
                else
                {
                    // op_inputs = CopySequenceSettingIndicesToNull(inputs, *unused_input_indices);
                }
            }
            else
                op_inputs = inputs;*/

            backwardFunction = backwardFunction ?? GetGradientFunction(op_name, inputs, attrs, results);
            TapeSetRecordOperation(op_name, inputs, results, input_ids, input_dtypes, backwardFunction);

            return true;
        }

        BackwardFunction GetGradientFunction(string op_name,
                     Tensor[] op_inputs,
                     object[] attrs,
                     Tensor[] op_outputs)
            => (out_grads, unneeded_gradients) =>
            {
                if(!ops.gradientFunctions.ContainsKey(op_name))
                {
                    throw new Exception($"gradientFunctions not find op_name: {op_name}");
                }

                if (ops.gradientFunctions[op_name] == null)
                    return new Tensor[op_inputs.Length];

                var oper = new EagerOperation
                {
                    Name = op_name,
                    NumInputs = op_inputs.Length,
                    Inputs = op_inputs,
                    NumOutputs = op_outputs.Length,
                    Outputs = op_outputs,
                    SkipInputIndices = unneeded_gradients,
                    Attrs = attrs
                };

                /*return op_name switch
                {
                    "Add" => math_grad._AddGrad(oper, out_grads),
                    "AddV2" => math_grad._AddV2Grad(oper, out_grads),
                    "BiasAdd" => nn_grad._BiasAddGrad(oper, out_grads),
                    "Cast" => math_grad._CastGrad(oper, out_grads),
                    "ConcatV2" => array_grad._ConcatV2Grad(oper, out_grads),
                    "Conv2D" => nn_grad._Conv2DGrad(oper, out_grads),
                    "ExpandDims" => array_grad._ExpandDimsGrad(oper, out_grads),
                    "Exp" => math_grad._ExpGrad(oper, out_grads),
                    "FusedBatchNormV3" => nn_grad._FusedBatchNormV3Grad(oper, out_grads),
                    "Id" => math_grad._IdGrad(oper, out_grads),
                    "LeakyRelu" => nn_grad._LeakyReluGrad(oper, out_grads),
                    "Log1p" => math_grad._Log1pGrad(oper, out_grads),
                    "Maximum" => math_grad._MaximumGrad(oper, out_grads),
                    "Mean" => math_grad._MeanGrad(oper, out_grads),
                    "Minimum" => math_grad._MinimumGrad(oper, out_grads),
                    "Mul" => math_grad._MulGrad(oper, out_grads),
                    "Neg" => math_grad._NegGrad(oper, out_grads),
                    "Pad" => array_grad._PadGrad(oper, out_grads),
                    "Pow" => math_grad._PowGrad(oper, out_grads),
                    "RealDiv" => math_grad._RealDivGrad(oper, out_grads),
                    "Read" => resource_variable_grad._ReadGrad(oper, out_grads),
                    "Reshape" => array_grad._ReshapeGrad(oper, out_grads),
                    "ResizeNearestNeighbor" => image_grad._ResizeNearestNeighborGrad(oper, out_grads),
                    "Select" => math_grad._SelectGrad(oper, out_grads),
                    "Sigmoid" => math_grad._SigmoidGrad(oper, out_grads),
                    "Sum" => math_grad._SumGrad(oper, out_grads),
                    "Sub" => math_grad._SubGrad(oper, out_grads),
                    "StridedSlice" => array_grad._StridedSliceGrad(oper, out_grads),
                    _ => ops.gradientFunctions[op_name](oper, out_grads)
                };*/

                return ops.gradientFunctions[op_name](oper, out_grads);
            };

        bool CouldForwardprop()
        {
            return HasAccumulator();
        }

        bool CouldBackprop()
        {
            return HasGradientTape();
        }
    }
}
