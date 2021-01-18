using Tensorflow.Eager;

namespace Tensorflow
{
    /// <summary>
    /// Represents a future for a read of a variable.
    /// Pretends to be the tensor if anyone looks.
    /// </summary>
    public class _UnreadVariable : BaseResourceVariable, IVariableV1
    {
        public override string Name => _in_graph_mode ? _parent_op.name : "UnreadVariable";

        public _UnreadVariable(Tensor handle, TF_DataType dtype, TensorShape shape,
            bool in_graph_mode, string unique_id)
        {
            _dtype = dtype;
            _shape = shape;
            base.handle = handle;
            _unique_id = unique_id;
            _in_graph_mode = in_graph_mode;

            if (handle is EagerTensor)
                _handle_name = "";
            else
                _handle_name = handle.name;
        }
    }
}
