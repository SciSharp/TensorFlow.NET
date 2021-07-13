namespace Tensorflow
{
    public class InitializerArgs
    {
        public string Name { get; set; }
        public Shape Shape { get; set; }
        public TF_DataType DType { get; set; }
        public bool VerifyShape { get; set; }

        public InitializerArgs(Shape shape,
            TF_DataType dtype = TF_DataType.DtInvalid,
            bool verify_shape = false,
            string name = null)
        {
            Shape = shape;
            DType = dtype;
            VerifyShape = verify_shape;
            Name = name;
        }
    }
}
