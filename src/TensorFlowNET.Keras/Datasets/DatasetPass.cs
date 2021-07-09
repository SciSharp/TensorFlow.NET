using Tensorflow.NumPy;

namespace Tensorflow.Keras.Datasets
{
    public class DatasetPass
    {
        public (NDArray, NDArray) Train { get; set; }
        public (NDArray, NDArray) Test { get; set; }

        public void Deconstruct(out NDArray x_train, out NDArray y_train, out NDArray x_test, out NDArray y_test)
        {
            x_train = Train.Item1;
            y_train = Train.Item2;
            x_test = Test.Item1;
            y_test = Test.Item2;
        }

        public void Deconstruct(out (NDArray, NDArray) train, out (NDArray, NDArray) test)
        {
            train = Train;
            test = Test;
        }
    }
}
