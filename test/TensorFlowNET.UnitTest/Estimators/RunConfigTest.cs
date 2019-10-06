using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Eager;
using Tensorflow.Estimators;

namespace TensorFlowNET.UnitTest.Estimators
{
    /// <summary>
    /// estimator/tensorflow_estimator/python/estimator/run_config_test.py
    /// </summary>
    [TestClass]
    public class RunConfigTest
    {
        private static readonly string _TEST_DIR = "test_dir";
        private static readonly string _MASTER = "master_";
        private static readonly string _NOT_SUPPORTED_REPLACE_PROPERTY_MSG = "Replacing .*is not supported";        
        private static readonly string _SAVE_CKPT_ERR = "`save_checkpoints_steps` and `save_checkpoints_secs` cannot be both set.";
        private static readonly string _MODEL_DIR_ERR = "model_dir should be non-empty";        
        private static readonly string _MODEL_DIR_TF_CONFIG_ERR = "model_dir in TF_CONFIG should be non-empty";
        private static readonly string _MODEL_DIR_MISMATCH_ERR = "`model_dir` provided in RunConfig construct, if set, must have the same value as the model_dir in TF_CONFIG. ";
        private static readonly string _SAVE_SUMMARY_STEPS_ERR = "save_summary_steps should be >= 0";        
        private static readonly string _SAVE_CKPT_STEPS_ERR = "save_checkpoints_steps should be >= 0";
        private static readonly string _SAVE_CKPT_SECS_ERR = "save_checkpoints_secs should be >= 0";
        private static readonly string _SESSION_CONFIG_ERR = "session_config must be instance of ConfigProto";
        private static readonly string _KEEP_CKPT_MAX_ERR = "keep_checkpoint_max should be >= 0";
        private static readonly string _KEEP_CKPT_HOURS_ERR = "keep_checkpoint_every_n_hours should be > 0";
        private static readonly string _TF_RANDOM_SEED_ERR = "tf_random_seed must be integer";
        private static readonly string _DEVICE_FN_ERR = "device_fn must be callable with exactly one argument \"op\".";
        private static readonly string _ONE_CHIEF_ERR = "The \"cluster\" in TF_CONFIG must have only one \"chief\" node.";
        private static readonly string _ONE_MASTER_ERR = "The \"cluster\" in TF_CONFIG must have only one \"master\" node.";
        private static readonly string _MISSING_CHIEF_ERR = "If \"cluster\" is set .* it must have one \"chief\" node";
        private static readonly string _MISSING_TASK_TYPE_ERR = "If \"cluster\" is set .* task type must be set";
        private static readonly string _MISSING_TASK_ID_ERR = "If \"cluster\" is set .* task index must be set";
        private static readonly string _INVALID_TASK_INDEX_ERR = "is not a valid task_id";
        private static readonly string _NEGATIVE_TASK_INDEX_ERR = "Task index must be non-negative number.";
        private static readonly string _INVALID_TASK_TYPE_ERR = "is not a valid task_type";
        private static readonly string _INVALID_TASK_TYPE_FOR_LOCAL_ERR = "If \"cluster\" is not set in TF_CONFIG, task type must be WORKER.";
        private static readonly string _INVALID_TASK_INDEX_FOR_LOCAL_ERR = "If \"cluster\" is not set in TF_CONFIG, task index must be 0.";
        private static readonly string _INVALID_EVALUATOR_IN_CLUSTER_WITH_MASTER_ERR = "If `master` node exists in `cluster`, task_type `evaluator` is not supported.";
        private static readonly string _INVALID_CHIEF_IN_CLUSTER_WITH_MASTER_ERR = "If `master` node exists in `cluster`, job `chief` is not supported.";
        private static readonly string _INVALID_SERVICE_TYPE_ERR = "If \"service\" is set in TF_CONFIG, it must be a dict. Given";
        private static readonly string _EXPERIMENTAL_MAX_WORKER_DELAY_SECS_ERR = "experimental_max_worker_delay_secs must be an integer if set.";
        private static readonly string _SESSION_CREATION_TIMEOUT_SECS_ERR = "session_creation_timeout_secs should be > 0";

        [TestMethod]
        public void test_default_property_values()
        {
            var config = new RunConfig();

            Assert.IsNull(config.model_dir);
            Assert.IsNull(config.session_config);
            Assert.IsNull(config.tf_random_seed);
            Assert.AreEqual(100, config.save_summary_steps);
            Assert.AreEqual(600, config.save_checkpoints_secs);
            Assert.AreEqual(5, config.keep_checkpoint_max);
            Assert.AreEqual(10000, config.keep_checkpoint_every_n_hours);
            Assert.IsNull(config.service);
            Assert.IsNull(config.device_fn);
            Assert.IsNull(config.experimental_max_worker_delay_secs);
            Assert.AreEqual(7200, config.session_creation_timeout_secs);
        }
    }
}
