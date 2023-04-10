using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Util
{
    internal static class function_utils
    {
        private static ByteString _rewriter_config_optimizer_disabled;
        public static ByteString get_disabled_rewriter_config()
        {
            if(_rewriter_config_optimizer_disabled is null)
            {
                var config = new ConfigProto();
                var rewriter_config = config.GraphOptions.RewriteOptions;
                rewriter_config.DisableMetaOptimizer = true;
                _rewriter_config_optimizer_disabled = config.ToByteString();
            }
            return _rewriter_config_optimizer_disabled;
        }
    }
}
