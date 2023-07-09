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
    List<IVariableV1>? _best_weights;
    CallbackParams _parameters;
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
        if (mode != "auto" && mode != "min" && mode != "max")
        {
            Console.WriteLine("EarlyStopping mode %s is unknown, fallback to auto mode.", mode);
        }
    }
    public void on_train_begin()
    {
        _wait = 0;
        _stopped_epoch = 0;
        _best_epoch = 0;
        _best = (float)np.Inf;
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
            _best_weights = _parameters.Model.Weights;
        }
        _wait += 1;

        if (_is_improvement(current, _best))
        {
            _best = current;
            _best_epoch = epoch;
            if (_restore_best_weights)
                _best_weights = _parameters.Model.TrainableWeights;
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
                _parameters.Model.Weights = _best_weights;
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
        bool less_op = (monitor_value - _min_delta) < reference_value;
        bool greater_op = (monitor_value - _min_delta) >= reference_value;
        if (_mode == "min")
            return less_op;
        else if (_mode == "max")
            return greater_op;
        else
        {
            if (_monitor.EndsWith("acc") || _monitor.EndsWith("accuracy") || _monitor.EndsWith("auc"))
            {
                return greater_op;
            }
            else
                return less_op;
        }
    }

    public void on_test_end(Dictionary<string, float> logs)
    {
    }
}
