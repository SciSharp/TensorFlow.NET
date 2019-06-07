# Chapter. Gradient

### Register custom gradient function

TF.NET is extensible which can be added custom gradient function.

```csharp
// define gradient function
ops.RegisterGradientFunction("ConcatV2", (oper, out_grads) => 
{
    var grad = grads[0];
    return new Tensor[]{ };    
});
```

