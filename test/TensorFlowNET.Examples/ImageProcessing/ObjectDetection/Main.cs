using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using Tensorflow.Contrib.Train;
using Tensorflow.Estimators;
using Tensorflow.Models.ObjectDetection;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.ImageProcessing.ObjectDetection
{
    public class Main : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = true;

        public string Name => "Object Detection API";

        ModelLib model_lib = new ModelLib();

        string model_dir = "D:/Projects/PythonLab/tf-models/research/object_detection/models/model";
        string pipeline_config_path = "ObjectDetection/Models/faster_rcnn_resnet101_voc07.config";
        int num_train_steps = 50;
        int sample_1_of_n_eval_examples = 1;
        int sample_1_of_n_eval_on_train_examples = 5;

        public bool Run()
        {
            var config = tf.estimator.RunConfig(model_dir: model_dir);

            var train_and_eval_dict = model_lib.create_estimator_and_inputs(run_config: config,
                hparams: new HParams(true),
                pipeline_config_path: pipeline_config_path,
                train_steps: num_train_steps,
                sample_1_of_n_eval_examples: sample_1_of_n_eval_examples,
                sample_1_of_n_eval_on_train_examples: sample_1_of_n_eval_on_train_examples);

            var estimator = train_and_eval_dict.estimator;
            var train_input_fn = train_and_eval_dict.train_input_fn;
            var eval_input_fns = train_and_eval_dict.eval_input_fns;
            var eval_on_train_input_fn = train_and_eval_dict.eval_on_train_input_fn;
            var predict_input_fn = train_and_eval_dict.predict_input_fn;
            var train_steps = train_and_eval_dict.train_steps;

            var (train_spec, eval_specs) = model_lib.create_train_and_eval_specs(train_input_fn,
                eval_input_fns,
                eval_on_train_input_fn,
                predict_input_fn,
                train_steps,
                eval_on_train_data: false);

            // Currently only a single Eval Spec is allowed.
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0]);

            return true;
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public void Predict(Session sess)
        {
            throw new NotImplementedException();
        }

        public void PrepareData()
        {
            throw new NotImplementedException();
        }

        public void Train(Session sess)
        {
            throw new NotImplementedException();
        }

        void IExample.Test(Session sess)
        {
            throw new NotImplementedException();
        }
    }
}
