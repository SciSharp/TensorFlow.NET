namespace Tensorflow.Operations
{
    public class FusedBatchNormParams
    {
        public string Name { get; set; }
        public Tensor YBackprop { get; set; }
        public Tensor X { get; set; }
        public Tensor Scale { get; set; }
        public Tensor ReserveSpace1 { get; set; }
        public Tensor ReserveSpace2 { get; set; }
        public Tensor ReserveSpace3 { get; set; }
        public float Epsilon { get; set; }
        public string DataFormat { get; set; }
        public bool IsTraining { get; set; }

        public FusedBatchNormParams()
        {
            Epsilon = 0.0001f;
            DataFormat = "NHWC";
            IsTraining = true;
        }
    }
}
