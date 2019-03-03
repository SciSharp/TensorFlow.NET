//Base classes for probability distributions.
using System;
using System.Collections.Generic;
using System.Text;


namespace Tensorflow
{
    abstract class _BaseDistribution : Object
    {
        // Abstract base class needed for resolving subclass hierarchy.
    }

    /// <summary>
    /// A generic probability distribution base class.
    /// Distribution is a base class for constructing and organizing properties
    /// (e.g., mean, variance) of random variables (e.g, Bernoulli, Gaussian). 
    /// </summary>
    class Distribution : _BaseDistribution
    {
        public TF_DataType _dtype {get;set;}
        public ReparameterizationType _reparameterization_type {get;set;}
        public bool _validate_args {get;set;}
        public bool _allow_nan_stats {get;set;}
        public Dictionary<object, object> _parameters  {get;set;}
        public List<object> _graph_parents  {get;set;}
        public string _name  {get;set;}

        /// <summary>
        /// Constructs the `Distribution'     
        /// **This is a private method for subclass use.**
        /// </summary>
        /// <param name="dtype"> The type of the event samples. `None` implies no type-enforcement.</param>
        /// <param name="reparameterization_type"> Instance of `ReparameterizationType`.
        /// If `distributions.FULLY_REPARAMETERIZED`, this `Distribution` can be reparameterized
        /// in terms of some standard distribution with a function whose Jacobian is constant for the support 
        /// of the standard distribution. If `distributions.NOT_REPARAMETERIZED`,
        /// then no such reparameterization is available.</param>
        /// <param name="validate_args"> When `True` distribution parameters are checked for validity despite
        /// possibly degrading runtime performance. When `False` invalid inputs silently render incorrect outputs.</param>
        /// <param name="allow_nan_stats"> When `True`, statistics (e.g., mean, mode, variance) use the value "`NaN`" 
        /// to indicate the result is undefined. When `False`, an exception is raised if one or more of the statistic's
        /// batch members are undefined.</param>
        /// <param name = "parameters"> `dict` of parameters used to instantiate this `Distribution`.</param>
        /// <param name = "graph_parents"> `list` of graph prerequisites of this `Distribution`.</param>
        /// <param name = "name"> Name prefixed to Ops created by this class. Default: subclass name.</param>
        /// <returns> Two `Tensor` objects: `mean` and `variance`.</returns>

        /* 
        private Distribution (
                TF_DataType dtype,
                ReparameterizationType reparameterization_type,
                bool validate_args,
                bool allow_nan_stats,
                Dictionary<object, object> parameters=null,
                List<object> graph_parents=null,
                string name= null)
                {
                    this._dtype = dtype;
                    this._reparameterization_type = reparameterization_type;
                    this._allow_nan_stats = allow_nan_stats;
                    this._validate_args = validate_args;
                    this._parameters = parameters;
                    this._graph_parents = graph_parents;
                    this._name = name;
                }
        */
    }

    /// <summary>
    /// Instances of this class represent how sampling is reparameterized.
    /// Two static instances exist in the distributions library, signifying
    /// one of two possible properties for samples from a distribution:
    /// `FULLY_REPARAMETERIZED`: Samples from the distribution are fully
    /// reparameterized, and straight-through gradients are supported.
    /// `NOT_REPARAMETERIZED`: Samples from the distribution are not fully
    /// reparameterized, and straight-through gradients are either partially
    /// unsupported or are not supported at all. In this case, for purposes of
    /// e.g. RL or variational inference, it is generally safest to wrap the
    /// sample results in a `stop_gradients` call and use policy
    /// gradients / surrogate loss instead.
    /// </summary>
    class ReparameterizationType
    {

    }


}