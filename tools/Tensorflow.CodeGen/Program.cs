using OneOf.Types;
using Protobuf.Text;
using System.Diagnostics;
using System.Text;
using System.Xml.Linq;
using Tensorflow.CodeGen;

GenOpsWriter writer = new(@"D:\development\tf.net\gen_ops_v2",
    @"D:\Apps\miniconda3\envs\tf2.11\Lib\site-packages\tensorflow\python\ops",
    @"D:\development\tf.net\tensorflow-2.11.0\tensorflow\core\api_def\base_api",
    @"D:\development\tf.net\tensorflow-2.11.0\tensorflow\core\ops\ops.pbtxt");

writer.WriteAll();
