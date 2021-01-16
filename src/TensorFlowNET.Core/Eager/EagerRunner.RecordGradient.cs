using System;
using System.Linq;
using Tensorflow.Gradients;
using static Tensorflow.Binding;
using static Tensorflow.tensorflow;

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
            tf.Logger.Debug($"RecordGradient: op_name={op_name}");

            Tensor[] op_outputs;
#pragma warning disable CS0219 // Variable is assigned but its value is never used
            bool op_outputs_tuple_created = false;
#pragma warning restore CS0219 // Variable is assigned but its value is never used
            var unused_output_indices = gradient_exclustions.OpGradientUnusedOutputIndices(op_name);
            if (unused_output_indices != null)
            {
                if (unused_output_indices.Length == 0)
                    op_outputs = new Tensor[0];
                else
                {
                    op_outputs_tuple_created = true;
                    // op_outputs = CopySequenceSettingIndicesToNull(results, *unused_output_indices);
                }
            }
            else
                op_outputs = results;

            Tensor[] op_inputs;
#pragma warning disable CS0219 // Variable is assigned but its value is never used
            bool op_inputs_tuple_created = false;
#pragma warning restore CS0219 // Variable is assigned but its value is never used
            var unused_input_indices = gradient_exclustions.OpGradientUnusedInputIndices(op_name);
            if (unused_input_indices != null)
            {
                if (unused_input_indices.Length == 0)
                    op_inputs = new Tensor[0];
                else
                {
                    op_inputs_tuple_created = true;
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

                var gradients = ops.gradientFunctions[op_name](new EagerOperation
                {
                    Name = op_name,
                    NumInputs = op_inputs.Length,
                    Inputs = op_inputs,
                    NumOutputs = op_outputs.Length,
                    Outputs = op_outputs,
                    SkipInputIndices = unneeded_gradients,
                    Attrs = attrs
                }, output_grads);

                return gradients;
            };

        bool CouldForwardprop()
        {
            return HasAccumulator();
        }

        bool CouldBackprop()
        {
            return HasGradientTape();
        }

        long[] MakeTensorIDList(Tensor[] tensors)
        {
            return tensors.Select(x => x.Id).ToArray();
        }

        TF_DataType[] MakeTensorDtypeList(Tensor[] tensors)
        {
            return tensors.Select(x => x.dtype).ToArray();
        }
    }
}
