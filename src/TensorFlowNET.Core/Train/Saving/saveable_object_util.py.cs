using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class saveable_object_util
    {
        /// <summary>
        /// Returns the variables and names that will be used for a Saver.
        /// </summary>
        /// <param name="names_to_saveables"></param>
        /// <returns></returns>
        public static SaveableObject[] validate_and_slice_inputs(RefVariable[] names_to_saveables)
        {
            var names_to_saveables_dict = op_list_to_dict(names_to_saveables);
            var saveables = new List<SaveableObject>();
            var seen_ops = new List<Tensor>();

            foreach (var item in names_to_saveables_dict)
            {
                foreach (var converted_saveable_object in saveable_objects_for_op(item.Value, item.Key))
                    _add_saveable(saveables, seen_ops, converted_saveable_object);
            }
            return saveables.ToArray();
        }

        private static void _add_saveable<T>(List<T> saveables, List<Tensor> seen_ops, T saveable) where T : SaveableObject
        {
            if (seen_ops.Contains(saveable.op))
                throw new ValueError($"The same saveable will be restored with two names: {saveable.name}");

            saveables.Add(saveable);
            seen_ops.Add(saveable.op);
        }

        /// <summary>
        /// Create `SaveableObject`s from an operation.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static IEnumerable<SaveableObject> saveable_objects_for_op(Tensor op, string name)
        {
            if (false)
            {

            }
            else
            {
                ops.init_scope();
                var variable = ops.internal_convert_to_tensor(op, as_ref: true);
                if (variable.op.type == "VariableV2")
                    yield return new ReferenceVariableSaveable(variable, "", name);
            }
        }

        public static Dictionary<string, Tensor> op_list_to_dict(RefVariable[] op_list, bool convert_variable_to_tensor = true)
        {
            op_list = op_list.OrderBy(x => x.name).ToArray();
            var names_to_saveables = new Dictionary<string, Tensor>();

            foreach(var var in op_list)
            {
                if (false)
                {
                    throw new NotImplementedException("op_list_to_dict");
                }
                else
                {
                    if(false) // eager
                    {

                    }
                    else
                    {
                        string name = "";
                        Tensor tensor = null;

                        if (convert_variable_to_tensor)
                        {
                            tensor = ops.internal_convert_to_tensor(var, as_ref: true);
                        }

                        if (var.op.type == "ReadVariableOp")
                            name = var.op.inputs[0].op.name;
                        else
                            name = var.op.name;

                        if (names_to_saveables.ContainsKey(name))
                            throw new ValueError($"At least two variables have the same name: {name}");

                        names_to_saveables[name] = tensor;
                    }
                }
            }

            return names_to_saveables;
        }
    }
}
