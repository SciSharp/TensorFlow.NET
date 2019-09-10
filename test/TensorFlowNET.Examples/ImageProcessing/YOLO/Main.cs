using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    /// <summary>
    /// Implementation of YOLO v3 object detector in Tensorflow
    /// https://github.com/YunYang1994/tensorflow-yolov3
    /// </summary>
    public class Main : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = false;
        public string Name => "YOLOv3";

        #region args
        Dictionary<int, string> classes;
        int num_classes;
        float learn_rate_init;
        float learn_rate_end;
        int first_stage_epochs;
        int second_stage_epochs;
        int warmup_periods;
        string time;
        float moving_ave_decay;
        int max_bbox_per_scale;
        int steps_per_period;

        Dataset trainset, testset;

        Config cfg;

        Tensor input_data;
        Tensor label_sbbox;
        Tensor label_mbbox;
        Tensor label_lbbox;
        Tensor true_sbboxes;
        Tensor true_mbboxes;
        Tensor true_lbboxes;
        Tensor trainable;

        Session sess;
        YOLOv3 model;
        VariableV1[] net_var;
        Tensor giou_loss, conf_loss, prob_loss;
        RefVariable global_step;
        Tensor learn_rate;
        Tensor loss;
        List<RefVariable> first_stage_trainable_var_list;
        #endregion

        public bool Run()
        {
            PrepareData();

            var graph = IsImportingGraph ? ImportGraph() : BuildGraph();

            var options = new SessionOptions();
            options.SetConfig(new ConfigProto { AllowSoftPlacement = true });
            using (var sess = tf.Session(graph, opts: options))
            {
                Train(sess);
            }

            return true;
        }

        public void Train(Session sess)
        {

        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            var graph = new Graph().as_default();

            tf_with(tf.name_scope("define_input"), scope =>
            {
                input_data = tf.placeholder(dtype: tf.float32, name: "input_data");
                label_sbbox = tf.placeholder(dtype: tf.float32, name: "label_sbbox");
                label_mbbox = tf.placeholder(dtype: tf.float32, name: "label_mbbox");
                label_lbbox = tf.placeholder(dtype: tf.float32, name: "label_lbbox");
                true_sbboxes = tf.placeholder(dtype: tf.float32, name: "sbboxes");
                true_mbboxes = tf.placeholder(dtype: tf.float32, name: "mbboxes");
                true_lbboxes = tf.placeholder(dtype: tf.float32, name: "lbboxes");
                trainable = tf.placeholder(dtype: tf.@bool, name: "training");
            });

            tf_with(tf.name_scope("define_loss"), scope =>
            {
                model = new YOLOv3(cfg, input_data, trainable);
                net_var = tf.global_variables();
                (giou_loss, conf_loss, prob_loss) = model.compute_loss(
                                                    label_sbbox, label_mbbox, label_lbbox,
                                                    true_sbboxes, true_mbboxes, true_lbboxes);
                loss = giou_loss + conf_loss + prob_loss;
            });

            Tensor global_step_update = null;
            tf_with(tf.name_scope("learn_rate"), scope =>
            {
                global_step = tf.Variable(1.0, dtype: tf.float64, trainable: false, name: "global_step");
                var warmup_steps = tf.constant(warmup_periods * steps_per_period,
                                        dtype: tf.float64, name: "warmup_steps");
                var train_steps = tf.constant((first_stage_epochs + second_stage_epochs) * steps_per_period,
                                        dtype: tf.float64, name: "train_steps");

                learn_rate = tf.cond(
                    pred: global_step < warmup_steps,
                    true_fn: delegate
                    {
                        return global_step / warmup_steps * learn_rate_init;
                    },
                    false_fn: delegate
                    {
                        return learn_rate_end + 0.5 * (learn_rate_init - learn_rate_end) *
                            (1 + tf.cos(
                                (global_step - warmup_steps) / (train_steps - warmup_steps) * Math.PI));
                    }
                );

                global_step_update = tf.assign_add(global_step, 1.0f);
            });

            Operation moving_ave = null;
            tf_with(tf.name_scope("define_weight_decay"), scope =>
            {
                var emv = tf.train.ExponentialMovingAverage(moving_ave_decay);
                var vars = tf.trainable_variables().Select(x => (RefVariable)x).ToArray();
                moving_ave = emv.apply(vars);
            });

            tf_with(tf.name_scope("define_first_stage_train"), scope =>
            {
                first_stage_trainable_var_list = new List<RefVariable>();
                foreach (var var in tf.trainable_variables())
                {
                    var var_name = var.op.name;
                    var var_name_mess = var_name.Split('/');
                    if (new[] { "conv_sbbox", "conv_mbbox", "conv_lbbox" }.Contains(var_name_mess[0]))
                        first_stage_trainable_var_list.Add(var as RefVariable);
                }

                var adam = tf.train.AdamOptimizer(learn_rate);
                var first_stage_optimizer = adam.minimize(loss, var_list: first_stage_trainable_var_list);
            });

            return graph;
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
            cfg = new Config(Name);

            string dataDir = Path.Combine(Name, "data");
            Directory.CreateDirectory(dataDir);

            classes = Utils.read_class_names(cfg.YOLO.CLASSES);
            num_classes = classes.Count;

            learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT;
            learn_rate_end = cfg.TRAIN.LEARN_RATE_END;
            first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS;
            second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS;
            warmup_periods = cfg.TRAIN.WARMUP_EPOCHS;
            DateTime now = DateTime.Now;
            time = $"{now.Year}-{now.Month}-{now.Day}-{now.Hour}-{now.Minute}-{now.Minute}";
            moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY;
            max_bbox_per_scale = 150;
            trainset = new Dataset("train", cfg);
            testset = new Dataset("test", cfg);
            steps_per_period = trainset.Length;
        }
    }
}
