using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Callbacks;

public class History : ICallback
{
    List<int> epochs;
    CallbackParams _parameters;
    public Dictionary<string, List<float>> history { get; set; }

    public History(CallbackParams parameters)
    {
        _parameters = parameters;
    }

    public void on_train_begin()
    {
        epochs = new List<int>();
        history = new Dictionary<string, List<float>>();
    }
    public void on_test_begin()
    {
        epochs = new List<int>();
        history = new Dictionary<string, List<float>>();
    }
    public void on_train_end() { }
    public void on_epoch_begin(int epoch)
    {

    }

    public void on_train_batch_begin(long step)
    {

    }

    public void on_train_batch_end(long end_step, Dictionary<string, float> logs)
    {
    }

    public void on_epoch_end(int epoch, Dictionary<string, float> epoch_logs)
    {
        epochs.Add(epoch);

        foreach (var log in epoch_logs)
        {
            if (!history.ContainsKey(log.Key))
            {
                history[log.Key] = new List<float>();
            }
            history[log.Key].Add(log.Value);
        }
    }

    public void on_predict_begin()
    {
        epochs = new List<int>();
        history = new Dictionary<string, List<float>>();
    }

    public void on_predict_batch_begin(long step)
    {

    }

    public void on_predict_batch_end(long end_step, Dictionary<string, Tensors> logs)
    {

    }

    public void on_predict_end()
    {

    }

    public void on_test_batch_begin(long step)
    {

    }

    public void on_test_batch_end(long end_step, Dictionary<string, float> logs)
    {
    }

    public void on_test_end(Dictionary<string, float> logs)
    {
    }
}
