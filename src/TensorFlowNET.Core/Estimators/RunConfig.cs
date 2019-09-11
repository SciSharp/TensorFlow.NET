﻿using System;

namespace Tensorflow.Estimators
{
    public class RunConfig
    {
        // A list of the property names in RunConfig that the user is allowed to change.
        private static readonly string[] _DEFAULT_REPLACEABLE_LIST = new []
        {
            "model_dir",
            "tf_random_seed",
            "save_summary_steps",
            "save_checkpoints_steps",
            "save_checkpoints_secs",
            "session_config",
            "keep_checkpoint_max",
            "keep_checkpoint_every_n_hours",
            "log_step_count_steps",
            "train_distribute",
            "device_fn",
            "protocol",
            "eval_distribute",
            "experimental_distribute",
            "experimental_max_worker_delay_secs",
            "session_creation_timeout_secs"
        };


        #region const values

        private const string _SAVE_CKPT_ERR = "`save_checkpoints_steps` and `save_checkpoints_secs` cannot be both set.";
        private const string _TF_CONFIG_ENV = "TF_CONFIG";
        private const string _TASK_ENV_KEY = "task";
        private const string _TASK_TYPE_KEY = "type";
        private const string _TASK_ID_KEY = "index";
        private const string _CLUSTER_KEY = "cluster";
        private const string _SERVICE_KEY = "service";
        private const string _SESSION_MASTER_KEY = "session_master";
        private const string _EVAL_SESSION_MASTER_KEY = "eval_session_master";
        private const string _MODEL_DIR_KEY = "model_dir";
        private const string _LOCAL_MASTER = "";
        private const string _GRPC_SCHEME = "grpc://";

        #endregion

        private static readonly object _USE_DEFAULT = new object();

        public string model_dir { get; set; }
        public ConfigProto session_config { get; set; }
        public int tf_random_seed { get; set; }        
        public int save_summary_steps { get; set; } = 100;
        public object save_checkpoints_steps { get; set; } = _USE_DEFAULT;
        public object save_checkpoints_secs { get; set; } = _USE_DEFAULT;
        public int keep_checkpoint_max { get; set; } = 5;        
        public int keep_checkpoint_every_n_hours { get; set; } = 10000;         
        public int log_step_count_steps{ get; set; } = 100;
        public object train_distribute { get; set; }
        public object device_fn { get; set; }
        public object protocol { get; set; }
        public object eval_distribute { get; set; }
        public object experimental_distribute { get; set; }
        public object experimental_max_worker_delay_secs { get; set; }
        public int session_creation_timeout_secs  { get; set; } = 7200;

        public RunConfig(string model_dir)
        {
            this.model_dir = model_dir;
        }
    }
}
