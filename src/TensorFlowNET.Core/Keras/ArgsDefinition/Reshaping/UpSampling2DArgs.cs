namespace Tensorflow.Keras.ArgsDefinition
{
    public class UpSampling2DArgs : LayerArgs
    {
        public Shape Size { get; set; }
        public string DataFormat { get; set; }
        /// <summary>
        /// 'nearest', 'bilinear'
        /// </summary>
        public string Interpolation { get; set; } = "nearest";
    }
}
