using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Tensorflow
{
    /// <summary>                                                                                                                                         <br></br>
    /// NDArray can be indexed using slicing                                                                                                              <br></br>
    /// A slice is constructed by start:stop:step notation                                                                                                <br></br>
    ///                                                                                                                                                   <br></br>
    /// Examples:                                                                                                                                         <br></br>
    ///                                                                                                                                                   <br></br>
    /// a[start:stop]  # items start through stop-1                                                                                                       <br></br>
    /// a[start:]      # items start through the rest of the array                                                                                        <br></br>
    /// a[:stop]       # items from the beginning through stop-1                                                                                          <br></br>
    ///                                                                                                                                                   <br></br>
    /// The key point to remember is that the :stop value represents the first value that is not                                                          <br></br>
    /// in the selected slice. So, the difference between stop and start is the number of elements                                                        <br></br>
    /// selected (if step is 1, the default).                                                                                                             <br></br>
    ///                                                                                                                                                   <br></br>
    /// There is also the step value, which can be used with any of the above:                                                                            <br></br>
    /// a[:]           # a copy of the whole array                                                                                                        <br></br>
    /// a[start:stop:step] # start through not past stop, by step                                                                                         <br></br>
    ///                                                                                                                                                   <br></br>
    /// The other feature is that start or stop may be a negative number, which means it counts                                                           <br></br>
    /// from the end of the array instead of the beginning. So:                                                                                           <br></br>
    /// a[-1]    # last item in the array                                                                                                                 <br></br>
    /// a[-2:]   # last two items in the array                                                                                                            <br></br>
    /// a[:-2]   # everything except the last two items                                                                                                   <br></br>
    /// Similarly, step may be a negative number:                                                                                                         <br></br>
    ///                                                                                                                                                   <br></br>
    /// a[::- 1]    # all items in the array, reversed                                                                                                    <br></br>
    /// a[1::- 1]   # the first two items, reversed                                                                                                       <br></br>
    /// a[:-3:-1]  # the last two items, reversed                                                                                                         <br></br>
    /// a[-3::- 1]  # everything except the last two items, reversed                                                                                      <br></br>
    ///                                                                                                                                                   <br></br>
    /// NumSharp is kind to the programmer if there are fewer items than                                                                                  <br></br>
    /// you ask for. For example, if you  ask for a[:-2] and a only contains one element, you get an                                                      <br></br>
    /// empty list instead of an error.Sometimes you would prefer the error, so you have to be aware                                                      <br></br>
    /// that this may happen.                                                                                                                             <br></br>
    ///                                                                                                                                                   <br></br>
    /// Adapted from Greg Hewgill's answer on Stackoverflow: https://stackoverflow.com/questions/509211/understanding-slice-notation                      <br></br>
    ///                                                                                                                                                   <br></br>
    /// Note: special IsIndex == true                                                                                                                     <br></br>
    /// It will pick only a single value at Start in this dimension effectively reducing the Shape of the sliced matrix by 1 dimension.                   <br></br>
    /// It can be used to reduce an N-dimensional array/matrix to a (N-1)-dimensional array/matrix                                                        <br></br>
    ///                                                                                                                                                   <br></br>
    /// Example:                                                                                                                                          <br></br>
    /// a=[[1, 2], [3, 4]]                                                                                                                                <br></br>
    /// a[:, 1] returns the second column of that 2x2 matrix as a 1-D vector                                                                              <br></br>
    /// </summary>
    public class Slice
    {
        /// <summary>
        /// return : for this dimension
        /// </summary>
        public static readonly Slice All = new Slice(null, null);

        /// <summary>
        /// return 0:0 for this dimension
        /// </summary>
        public static readonly Slice None = new Slice(0, 0, 1);

        /// <summary>
        /// fill up the missing dimensions with : at this point, corresponds to ... 
        /// </summary>
        public static readonly Slice Ellipsis = new Slice(0, 0, 1) { IsEllipsis = true };

        /// <summary>
        /// insert a new dimension at this point
        /// </summary>
        public static readonly Slice NewAxis = new Slice(0, 0, 1) { IsNewAxis = true };

        /// <summary>
        /// return exactly one element at this dimension and reduce the shape from n-dim to (n-1)-dim
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public static Slice Index(int index) => new Slice(index, index + 1) { IsIndex = true };

        ///// <summary>
        ///// return multiple elements for this dimension specified by the given index array (or boolean mask array)
        ///// </summary>
        ///// <param name="index_array_or_mask"></param>
        ///// <returns></returns>
        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        //public static Slice Select(NDArray index_array_or_mask) => new Slice(null, null) { Selection=index_array_or_mask };

        public int? Start;
        public int? Stop;
        public int Step;
        public bool IsIndex;
        public bool IsEllipsis;
        public bool IsNewAxis;

        ///// <summary>
        ///// Array of integer indices to select elements by index extraction or boolean values to select by masking the elements of the given dimension.
        ///// </summary>
        //public NDArray Selection = null;

        /// <summary>
        /// Length of the slice. 
        /// <remarks>
        /// The length is not guaranteed to be known for i.e. a slice like ":". Make sure to check Start and Stop 
        /// for null before using it</remarks>
        /// </summary>
        public int? Length => Stop - Start;

        /// <summary>
        /// ndarray can be indexed using slicing
        /// slice is constructed by start:stop:step notation
        /// </summary>
        /// <param name="start">Start index of the slice, null means from the start of the array</param>
        /// <param name="stop">Stop index (first index after end of slice), null means to the end of the array</param>
        /// <param name="step">Optional step to select every n-th element, defaults to 1</param>
        public Slice(int? start = null, int? stop = null, int step = 1, bool isIndex = false)
        {
            Start = start;
            Stop = stop;
            Step = step;
            IsIndex = isIndex; 
        }

        public Slice(string slice_notation)
        {
            Parse(slice_notation);
        }

        /// <summary>
        /// Parses Python array slice notation and returns an array of Slice objects
        /// </summary>
        public static Slice[] ParseSlices(string multi_slice_notation)
        {
            return Regex.Split(multi_slice_notation, @",\s*").Where(s => !string.IsNullOrWhiteSpace(s)).Select(token => new Slice(token)).ToArray();
        }

        /// <summary>
        /// Creates Python array slice notation out of an array of Slice objects (mainly used for tests)
        /// </summary>
        public static string FormatSlices(params Slice[] slices)
        {
            return string.Join(",", slices.Select(s => s.ToString()));
        }

        private void Parse(string slice_notation)
        {
            if (string.IsNullOrEmpty(slice_notation))
                throw new ArgumentException("Slice notation expected, got empty string or null");
            var match = Regex.Match(slice_notation, @"^\s*((?'start'[+-]?\s*\d+)?\s*:\s*(?'stop'[+-]?\s*\d+)?\s*(:\s*(?'step'[+-]?\s*\d+)?)?|(?'index'[+-]?\s*\d+)|(?'ellipsis'\.\.\.)|(?'newaxis'(np\.)?newaxis))\s*$");
            if (!match.Success)
                throw new ArgumentException($"Invalid slice notation: '{slice_notation}'");
            if (match.Groups["ellipsis"].Success)
            {
                Start = 0;
                Stop = 0;
                Step = 1;
                IsEllipsis = true;
                return;
            }
            if (match.Groups["newaxis"].Success)
            {
                Start = 0;
                Stop = 0;
                Step = 1;
                IsNewAxis = true;
                return;
            }
            if (match.Groups["index"].Success)
            {
                if (!int.TryParse(Regex.Replace(match.Groups["index"].Value ?? "", @"\s+", ""), out var start))
                    throw new ArgumentException($"Invalid value for index: '{match.Groups["index"].Value}'");
                Start = start;
                Stop = start + 1;
                Step = 1; // special case for dimensionality reduction by picking a single element
                IsIndex = true;
                return;
            }
            var start_string = Regex.Replace(match.Groups["start"].Value ?? "", @"\s+", ""); // removing spaces from match to be able to parse what python allows, like: "+ 1" or  "-   9";
            var stop_string = Regex.Replace(match.Groups["stop"].Value ?? "", @"\s+", "");
            var step_string = Regex.Replace(match.Groups["step"].Value ?? "", @"\s+", "");

            if (string.IsNullOrWhiteSpace(start_string))
                Start = null;
            else
            {
                if (!int.TryParse(start_string, out var start))
                    throw new ArgumentException($"Invalid value for start: {start_string}");
                Start = start;
            }

            if (string.IsNullOrWhiteSpace(stop_string))
                Stop = null;
            else
            {
                if (!int.TryParse(stop_string, out var stop))
                    throw new ArgumentException($"Invalid value for start: {stop_string}");
                Stop = stop;
            }

            if (string.IsNullOrWhiteSpace(step_string))
                Step = 1;
            else
            {
                if (!int.TryParse(step_string, out var step))
                    throw new ArgumentException($"Invalid value for start: {step_string}");
                Step = step;
            }
        }

        #region Equality comparison

        public static bool operator ==(Slice a, Slice b)
        {
            if (ReferenceEquals(a, b))
                return true;

            if (a is null || b is null)
                return false;

            return a.Start == b.Start && a.Stop == b.Stop && a.Step == b.Step;
        }

        public static bool operator !=(Slice a, Slice b)
        {
            return !(a == b);
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;

            if (obj.GetType() != typeof(Slice))
                return false;

            var b = (Slice)obj;
            return Start == b.Start && Stop == b.Stop && Step == b.Step;
        }

        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }

        #endregion

        public override string ToString()
        {
            if (IsIndex)
                return $"{Start ?? 0}";
            else if (IsNewAxis)
                return "np.newaxis";
            else if (IsEllipsis)
                return "...";
            var optional_step = Step == 1 ? "" : $":{Step}";
            return $"{(Start == 0 ? "" : Start.ToString())}:{(Stop == null ? "" : Stop.ToString())}{optional_step}";
        }

        // return the size of the slice, given the data dimension on this axis
        // note: this works only with sanitized shapes!
        public int GetSize()
        {
            var astep = Math.Abs(Step);
            return (Math.Abs(Start.Value - Stop.Value) + (astep - 1)) / astep;
        }

        #region Operators

        public static Slice operator ++(Slice a)
        {
            if (a.Start.HasValue)
                a.Start++;
            if (a.Stop.HasValue)
                a.Stop++;
            return a;
        }

        public static Slice operator --(Slice a)
        {
            if (a.Start.HasValue)
                a.Start--;
            if (a.Stop.HasValue)
                a.Stop--;
            return a;
        }

        public static implicit operator Slice(int index) => Slice.Index(index);
        public static implicit operator Slice(string slice) => new Slice(slice);
        //public static implicit operator Slice(NDArray selection) => Slice.Select(selection);

        #endregion
    }
}
