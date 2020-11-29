namespace Tensorflow.Keras.ArgsDefinition
{
    public class UpSampling2DArgs : LayerArgs
    {
        public TensorShape Size { get; set; }
        public string DataFormat { get; set; }
        /// <summary>
        /// 'nearest', 'bilinear'
        /// </summary>
        public string Interpolation { get; set; } = "nearest";
    }
}
