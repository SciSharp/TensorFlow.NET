using Protobuf.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tensorflow.CodeGen
{
    public class GenOpsWriter
    {
        private string _basePath;
        private Dictionary<string, OpDef> _opMap;
        private OpClassifier _opClassifier;
        private FunctionGenerator _g = new();

        public GenOpsWriter(string basePath, string pythonFilesDirectory, string opDefFilename)
        {
            _basePath = basePath;

            var opDefs = ReadAllOpDefs(opDefFilename);
            _opMap = opDefs.Op.ToDictionary(
                x => Tensorflow.CodeGen.Utils.ConvertToUnderscore(x.Name), x => x);
            _opClassifier = new OpClassifier(pythonFilesDirectory);
        }

        public void WriteAll()
        {
            foreach(var (target, set) in _opClassifier.OpSet)
            {
                StringBuilder sb = new StringBuilder();

                // Write file header.
                sb.AppendLine("/*Wrappers around TensorFlow ops. This file is MACHINE GENERATED! Do not edit.*/");
                sb.AppendLine();

                // Add commonly used namespaces.
                sb.AppendLine("using Tensorflow.Eager;");
                sb.AppendLine("using Tensorflow.Contexts;");
                sb.AppendLine("using static Tensorflow.Binding;");
                sb.AppendLine();

                // Specify the namespace
                sb.AppendLine("namespace Tensorflow;");
                sb.AppendLine();

                // Write class name
                sb.AppendLine($"internal static class {target}");
                sb.AppendLine("{");

                foreach(var funcName in set)
                {
                    if(_opMap.ContainsKey(funcName))
                    {
                        var opDef = _opMap[funcName];
                        _g.AppendFunction(opDef, sb);
                    }
                    else if (funcName.StartsWith("_"))
                    {
                        var opDef = _opMap[funcName.Substring(1)];
                        _g.AppendFunction(opDef, sb);
                    }
                }

                // Close class scope.
                sb.AppendLine("}");

                string fullFilePath = Path.Combine(_basePath, $"{target}.cs");
                File.WriteAllText(fullFilePath, sb.ToString());
            }
        }

        private OpList ReadAllOpDefs(string path)
        {
            var text = File.ReadAllText(path);
            var opDefs = OpList.Parser.ParseText(text);
            return opDefs;
        }
    }
}
