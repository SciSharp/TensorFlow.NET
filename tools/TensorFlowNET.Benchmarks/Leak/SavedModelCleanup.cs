using BenchmarkDotNet.Attributes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace Tensorflow.Benchmark.Leak
{
	/// <summary>
	/// https://github.com/SciSharp/TensorFlow.NET/issues/418
	/// </summary>
	public class SavedModelCleanup
	{
		[Benchmark]
		public void Run()
		{
			var modelDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			var ClassifierModelPath = Path.Combine(modelDir, "Leak", "TestModel", "saved_model");

			for (var i = 0; i < 1024; i++)
			{
				var sess = Session.LoadFromSavedModel(ClassifierModelPath);
				var g = sess.graph.as_default();
				var inputOp = g.OperationByName("inference_input");
				var outputOp = g.OperationByName("StatefulPartitionedCall");

				var inp = np.zeros(new Shape(new int[] { 1, 2, 96 }), TF_DataType.TF_FLOAT);
				sess.run(outputOp.outputs[0], new FeedItem(inputOp.outputs[0], inp));
			}
		}
	}
}
