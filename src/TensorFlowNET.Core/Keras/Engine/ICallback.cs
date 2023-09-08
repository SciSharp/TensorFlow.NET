namespace Tensorflow.Keras.Engine;

public interface ICallback
{
    Dictionary<string, List<float>> history { get; set; }
    void on_train_begin();
    void on_train_end();
    void on_epoch_begin(int epoch);
    void on_train_batch_begin(long step);
    void on_train_batch_end(long end_step, Dictionary<string, float> logs);
    void on_epoch_end(int epoch, Dictionary<string, float> epoch_logs);
    void on_predict_begin();
    void on_predict_batch_begin(long step);
    void on_predict_batch_end(long end_step, Dictionary<string, Tensors> logs);
    void on_predict_end();
    void on_test_begin();
    void on_test_end(Dictionary<string, float> logs);
    void on_test_batch_begin(long step);
    void on_test_batch_end(long end_step, Dictionary<string, float> logs);


}
