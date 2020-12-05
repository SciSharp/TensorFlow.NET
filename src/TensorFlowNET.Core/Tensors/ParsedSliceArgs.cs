using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class ParsedSliceArgs
    {
        public int[] Begin { get; set; }
        public Tensor PackedBegin { get; set; }
        public int[] End { get; set; }
        public Tensor PackedEnd { get; set; }
        public int[] Strides { get; set; }
        public Tensor PackedStrides { get; set; }
        public int BeginMask { get; set; }
        public int EndMask { get; set; }
        public int ShrinkAxisMask { get; set; }
        public int NewAxisMask { get; set; }
        public int EllipsisMask { get; set; }
    }
}
