using BenchmarkDotNet.Attributes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Tensorflow.Benchmark.Leak
{
	
	public class SavedModelCleanup
	{
		[Benchmark]
		public void Run()
		{
			var modelDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			var ClassifierModelPath = Path.Combine(modelDir, "Leak", "TestModel", "saved_model");

			for (var i = 0; i < 50; i++)
			{
				var session = Session.LoadFromSavedModel(ClassifierModelPath);

				session.graph.Exit();
				session.graph.Dispose();
				session.Dispose();
			}
		}
	}
}
