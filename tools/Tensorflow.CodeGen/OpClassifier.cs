using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;

namespace Tensorflow.CodeGen
{
    public class OpClassifier
    {
        private static readonly string _filenamePattern = @"^gen_[a-z_]*_ops.py$";
        private static readonly string _pythonFunctionPattern = @"def\s+(\w+\d*\w*)\((?:\s*\w+\s*(?:=\s*[\S]*)*,\s*)*\s*name=None\):";
        private Dictionary<string, HashSet<string>> _opSet = new();
        public Dictionary<string, HashSet<string>> OpSet => _opSet;
        public OpClassifier(string pythonFileFolder, IEnumerable<string> funcNames)
        {
            DirectoryInfo directory = new DirectoryInfo(pythonFileFolder);

            Dictionary<string, string> fileContentMap = new();
            foreach (FileInfo file in directory.GetFiles())
            {
                if (Regex.IsMatch(file.Name, _filenamePattern))
                {
                    Console.WriteLine(file.Name);
                    string filenamePrefix = file.Name.Split('.')[0];
                    string content = File.ReadAllText(file.FullName);
                    fileContentMap[filenamePrefix] = content;
                }
            }

            foreach(var funcName in funcNames)
            {
                Console.WriteLine(funcName);
                string funcPattern = @$"^def\s+{funcName}\(";
                string fallbackFuncPattern = @$"^def\s+{funcName}_eager_fallback\(";
                foreach (var (target, content) in fileContentMap)
                {
                    if(content.Contains($"def {funcName}") && content.Contains($"def {funcName}_eager_fallback"))
                    {
                        _opSet.SetDefault(target, new HashSet<string>()).Add(funcName);
                    }
                    else if (content.Contains($"def _{funcName}") && content.Contains($"def _{funcName}_eager_fallback"))
                    {
                        _opSet.SetDefault(target, new HashSet<string>()).Add(funcName);
                    }
                }
            }
        }
    }
}
