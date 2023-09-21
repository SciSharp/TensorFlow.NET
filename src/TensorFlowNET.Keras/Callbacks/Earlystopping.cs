using Tensorflow.Keras.Engine;
namespace Tensorflow.Keras.Callbacks;


/// <summary>
/// Stop training when a monitored metric has stopped improving. 
/// </summary>
public class EarlyStopping: ICallback
{
    int _paitence;
    float _min_delta;
    int _verbose;
    int _stopped_epoch;
    int _wait;
    int _best_epoch;
    int _start_from_epoch;
    float _best;
    float _baseline;
    string _monitor;
    string _mode;
    bool _restore_best_weights;
    List<NDArray>? _best_weights;
    CallbackParams _parameters;
    Func<NDArray, NDArray, NDArray> _monitor_op;

    public Dictionary<string, List<float>>? history { get; set; }
    // user need to pass a CallbackParams to EarlyStopping, CallbackParams at least need the model
    public EarlyStopping(CallbackParams parameters,string monitor = "val_loss", float min_delta = 0f, int patience = 0,
        int verbose = 1, string mode = "auto", float baseline = 0f, bool restore_best_weights = false,
        int start_from_epoch = 0)
    {
        _parameters = parameters;
        _stopped_epoch = 0;
        _wait = 0;
        _monitor = monitor;
        _paitence = patience;
        _verbose = verbose;
        _baseline = baseline;
        _start_from_epoch = start_from_epoch;
        _min_delta = Math.Abs(min_delta);
        _restore_best_weights = restore_best_weights;
        _mode = mode;

        if (_mode != "auto" && _mode != "min" && _mode != "max")
        {
            Console.WriteLine($"EarlyStopping mode {_mode} is unknown, fallback to auto mode.");
            _mode = "auto";
        }

        if (_mode == "min")
        {
            _monitor_op = np.less;
        }
        else if (_mode == "max")
        {
            _monitor_op = np.greater;
        }
        else
        {
            if (_monitor.EndsWith("acc") || _monitor.EndsWith("accuracy") || _monitor.EndsWith("auc"))
            {
                _monitor_op = np.greater;
            }
            else
            {
                _monitor_op = np.less;
            }   
        }

        if (_monitor_op == np.greater)
        {
            _min_delta *= 1;
        }
        else
        {
            _min_delta *= -1;
        }
    }
    public void on_train_begin()
    {
        _wait = 0;
        _stopped_epoch = 0;
        _best = _monitor_op == np.less ? (float)np.Inf : (float)-np.Inf;
        _best_weights = null;
        _best_epoch = 0;
    }

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
        var current = get_monitor_value(epoch_logs);
        // If no monitor value exists or still in initial warm-up stage.
        if (current == 0f || epoch < _start_from_epoch)
            return;
        // Restore the weights after first epoch if no progress is ever made.
        if (_restore_best_weights && _best_weights == null)
        {
            _best_weights = _parameters.Model.get_weights();
        }
        _wait += 1;

        if (_is_improvement(current, _best))
        {
            _best = current;
            _best_epoch = epoch;
            if (_restore_best_weights)
                _best_weights = _parameters.Model.get_weights();
            // Only restart wait if we beat both the baseline and our previous best.
            if (_baseline == 0f || _is_improvement(current, _baseline))
                _wait = 0;
        }
        // Only check after the first epoch.
        if (_wait >= _paitence && epoch > 0)
        {
            _stopped_epoch = epoch;
            _parameters.Model.Stop_training = true;
            if (_restore_best_weights && _best_weights != null)
            {
                if (_verbose > 0)
                {
                    Console.WriteLine($"Restoring model weights from the end of the best epoch: {_best_epoch + 1}");
                }
                _parameters.Model.set_weights(_best_weights);
            }
        }
    }
    public void on_train_end()
    {
        if (_stopped_epoch > 0 && _verbose > 0)
        {
            Console.WriteLine($"Epoch {_stopped_epoch + 1}: early stopping");
        }
    }
    public void on_predict_begin() { }
    public void on_predict_batch_begin(long step) { }
    public void on_predict_batch_end(long end_step, Dictionary<string, Tensors> logs) { }
    public void on_predict_end() { }
    public void on_test_begin() { }
    public void on_test_batch_begin(long step) { }
    public void on_test_batch_end(long end_step, Dictionary<string, float> logs) { }

    float get_monitor_value(Dictionary<string, float> logs)
    {
        logs = logs ?? new Dictionary<string, float>();
        float monitor_value = logs[_monitor];
        if (monitor_value == 0f)
        {
            Console.WriteLine($"Early stopping conditioned on metric {_monitor} " +
                $"which is not available. Available metrics are: {string.Join(", ", logs.Keys)}");
        }
        return monitor_value;
    }
    public bool _is_improvement(float monitor_value, float reference_value)
    {
        return _monitor_op(monitor_value - _min_delta, reference_value);
    }

    public void on_test_end(Dictionary<string, float> logs)
    {
    }
}
