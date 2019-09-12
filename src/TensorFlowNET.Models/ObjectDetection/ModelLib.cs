using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Estimators;
using System.Linq;
using Tensorflow.Contrib.Train;
using Tensorflow.Models.ObjectDetection.Utils;

namespace Tensorflow.Models.ObjectDetection
{
    public class ModelLib
    {
        public TrainAndEvalDict create_estimator_and_inputs(RunConfig run_config,
            HParams hparams = null,
            string pipeline_config_path = null,
            int train_steps = 0,
            int sample_1_of_n_eval_examples = 0,
            int sample_1_of_n_eval_on_train_examples = 1)
        {
            var config = ConfigUtil.get_configs_from_pipeline_file(pipeline_config_path);
            var eval_input_configs = config.EvalInputReader;

            var eval_input_fns = new Action[eval_input_configs.Count];
            var eval_input_names = eval_input_configs.Select(eval_input_config => eval_input_config.Name).ToArray();
            Action model_fn = () => { };
            var estimator = tf.estimator.Estimator(model_fn: model_fn, config: run_config);

            return new TrainAndEvalDict
            {
                estimator = estimator,
                train_steps = train_steps,
                eval_input_fns = eval_input_fns,
                eval_input_names = eval_input_names
            };
        }

        public (TrainSpec, EvalSpec[]) create_train_and_eval_specs(Action train_input_fn, Action[] eval_input_fns, Action eval_on_train_input_fn, 
            Action predict_input_fn, int train_steps, bool eval_on_train_data = false, 
            string final_exporter_name = "Servo", string[] eval_spec_names = null)
        {
            var train_spec = tf.estimator.TrainSpec(input_fn: train_input_fn, max_steps: train_steps);

            if (eval_spec_names == null)
                eval_spec_names = range(len(eval_input_fns))
                    .Select(x => x.ToString())
                    .ToArray();

            var eval_specs = new List<EvalSpec>()
            {
                new EvalSpec("", null, null) // for test.
            };
            foreach (var (index, (eval_spec_name, eval_input_fn)) in enumerate(zip(eval_spec_names, eval_input_fns).ToList()))
            {
                var exporter_name = index == 0 ? final_exporter_name : $"{final_exporter_name}_{eval_spec_name}";
                var exporter = tf.estimator.FinalExporter(name: exporter_name, serving_input_receiver_fn: predict_input_fn);
                eval_specs.Add(tf.estimator.EvalSpec(name: eval_spec_name,
                    input_fn: eval_input_fn,
                    exporters: exporter));
            }

            if (eval_on_train_data)
                throw new NotImplementedException("");

            return (train_spec, eval_specs.ToArray());
        }
    }
}
