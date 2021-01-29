using Tensorflow.Keras.Preprocessings;

namespace Tensorflow.Keras
{
    public partial class Preprocessing
    {
        public Sequence sequence => new Sequence();
        public DatasetUtils dataset_utils => new DatasetUtils();
        public TextApi text => _text;

        private static TextApi _text = new TextApi();
    }
}
