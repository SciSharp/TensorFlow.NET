namespace Tensorflow.Framework.Models
{
    /// <summary>
    /// Describes a dense object with shape, dtype, and name.
    /// </summary>
    public class DenseSpec : TypeSpec
    {
        protected TensorShape _shape;
        public TensorShape shape => _shape;

        protected TF_DataType _dtype;
        public TF_DataType dtype => _dtype;

        protected string _name;
        public string name => _name;

        public DenseSpec(TensorShape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            _shape = shape;
            _dtype = dtype;
            _name = name;
        }

        public override string ToString()
            => $"shape={_shape}, dtype={_dtype.as_numpy_name()}, name={_name}";
    }
}
