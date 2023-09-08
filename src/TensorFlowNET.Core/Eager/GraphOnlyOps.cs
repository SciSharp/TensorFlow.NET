using Tensorflow;

internal static class GraphOnlyOps
{
    /// <summary>
    /// Graph-only version of tf.compat.v1.placeholder(), for internal use only.
    /// </summary>
    /// <param name="dtyype"></param>
    /// <param name="shape"></param>
    /// <param name="name"></param>
    /// <returns></returns>
    internal static Tensor graph_placeholder(TF_DataType dtype, Shape shape, string? name = null)
    {
        var dtype_value = new AttrValue() { Type = dtype.as_datatype_enum() };
        var shape_value = new AttrValue() { Shape = shape.as_proto() };
        var g = ops.get_default_graph();
        Dictionary<string, AttrValue> attrs = new();
        attrs["dtype"] = dtype_value;
        attrs["shape"] = shape_value;
        var op = g.create_op("Placeholder", new Tensor[0], new TF_DataType[] { dtype },
            new TF_DataType[0], attrs: attrs, name: name);
        var result = op.outputs[0];
        return result;
    }
}