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
            Func<BackwardFunction> getBackwardFunction = null)
        {
            bool should_record = ShouldRecord(inputs);

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
            tf.Logger.Debug($"RecordGradient: op_name={op_name}");

            Tensor[] op_outputs;
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

            Tensor[] op_inputs;
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
                op_inputs = inputs;

            TapeSetRecordOperation(op_name, inputs, results,
                getBackwardFunction ?? GetBackwradFunction(op_name, inputs, attrs, results));

            return true;
        }

        Func<BackwardFunction> GetBackwradFunction(string op_name,
                     Tensor[] op_inputs,
                     object[] attrs,
                     Tensor[] op_outputs)
        {
            return () => GetGradientFunction(op_name, op_inputs, attrs, op_outputs);
        }

        BackwardFunction GetGradientFunction(string op_name,
                     Tensor[] op_inputs,
                     object[] attrs,
                     Tensor[] op_outputs)
            => (output_grads, unneeded_gradients) =>
            {
                if (ops.gradientFunctions[op_name] == null)
                    return new Tensor[op_inputs.Length];

                var op = new EagerOperation
                {
                    Name = op_name,
                    NumInputs = op_inputs.Length,
                    Inputs = op_inputs,
                    NumOutputs = op_outputs.Length,
                    Outputs = op_outputs,
                    SkipInputIndices = unneeded_gradients,
                    Attrs = attrs
                };

                return ops.gradientFunctions[op_name](op, output_grads);
            };

        bool CouldForwardprop()
        {
            return HasAccumulator();
        }

        bool CouldBackprop()
        {
            return HasGradientTape();
        }

        TF_DataType[] MakeTensorDtypeList(Tensor[] tensors)
        {
            return tensors.Select(x => x.dtype).ToArray();
        }
    }
}
