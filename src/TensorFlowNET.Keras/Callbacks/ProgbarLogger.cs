using System.Diagnostics;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Callbacks
{
    public class ProgbarLogger : ICallback
    {
        bool _called_in_fit = false;
        int seen = 0;
        CallbackParams _parameters;
        Stopwatch _sw;

        public Dictionary<string, List<float>> history { get; set; }

        public ProgbarLogger(CallbackParams parameters)
        {
            _parameters = parameters;
        }

        public void on_train_begin()
        {
            _called_in_fit = true;
            _sw = new Stopwatch();
        }
        public void on_train_end() { }
        public void on_test_begin()
        {
            _sw = new Stopwatch();
        }
        public void on_epoch_begin(int epoch)
        {
            _reset_progbar();
            _maybe_init_progbar();
            Binding.tf_output_redirect.WriteLine($"Epoch: {epoch + 1:D3}/{_parameters.Epochs:D3}");
        }

        public void on_train_batch_begin(long step)
        {
            _sw.Restart();
        }

        public void on_train_batch_end(long end_step, Dictionary<string, float> logs)
        {
            _sw.Stop();
            var elapse = _sw.ElapsedMilliseconds;
            var results = string.Join(" - ", logs.Select(x => $"{x.Key}: {(float)x.Value:F6}"));

            var progress = "";
            var length = 30.0 / _parameters.Steps;
            for (int i = 0; i < Math.Floor(end_step * length - 1); i++)
                progress += "=";
            if (progress.Length < 28)
                progress += ">";
            else
                progress += "=";

            var remaining = "";
            for (int i = 1; i < 30 - progress.Length; i++)
                remaining += ".";

            Binding.tf_output_redirect.Write($"{end_step + 1:D4}/{_parameters.Steps:D4} [{progress}{remaining}] - {elapse}ms/step - {results}");
            if (!Console.IsOutputRedirected)
            {
                Console.CursorLeft = 0;
            }
        }

        public void on_epoch_end(int epoch, Dictionary<string, float> epoch_logs)
        {
            Console.WriteLine();
        }

        void _reset_progbar()
        {
            seen = 0;
        }

        void _maybe_init_progbar()
        {

        }

        public void on_predict_begin()
        {
            _reset_progbar();
            _maybe_init_progbar();
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
            _sw.Restart();
        }
        public void on_test_batch_end(long end_step, Dictionary<string, float> logs)
        {
            _sw.Stop();
            var elapse = _sw.ElapsedMilliseconds;
            var results = string.Join(" - ", logs.Select(x => $"{x.Key}: {x.Value:F6}"));

            Binding.tf_output_redirect.Write($"{end_step + 1:D4}/{_parameters.Steps:D4} - {elapse}ms/step - {results}");
            if (!Console.IsOutputRedirected)
            {
                Console.CursorLeft = 0;
            }
        }

        public void on_test_end(Dictionary<string, float> logs)
        {
        }
    }
}
