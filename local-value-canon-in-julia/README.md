---
title: "Local Value Canonicalization in Julia"
date: 2021-08-21T20:15:57+08:00
draft: false
---

This is an article explaining my interpretation and implementation of **local value numbering**, a compiler optimization technique.

## Foreword

In this article, I will first introduce local value numbering, then explain my interpretation of the algorithm.  
Finally, I will explain my how I implemented the algorithm in Julia.  
The language I am optimizing is [Bril](https://github.com/sampsyo/bril), an educational compiler IR designed for the compiler course [CS 6120](https://www.cs.cornell.edu/courses/cs6120/2020fa/).

## Local Value Numbering

**Local value numbering** (LVN) is an optimization technique targeting a block of code without branching (i.e. an if else block). 
LVN optimizes the code such that **every value is assigned to only one variable**, aka the canonical variable. Consider the following example:
```
@main {
    a: int = const 39;
    b: int = const 39;
}
```
After parsing the first line, LVN assigns the value 39 to the variable `a`. 
Thus, when LVN sees that variable `b` is also assigned to the value 39, it considers `b` as redundant.
The key insight is that LVN separates values (39) from variables (`a`, `b`), and replaces all references to the same value with its canonical variable.  

LVN separates values from variables by **numbering the values** and make variables refer to the numbers. 
In this case, we would number the value 39 as 1, and variables `a` and `b` both refer to the number 1. 
By doing this, we can optimize future references to `b`. This can be demonstrated by a more complex example:
```
@main {
    a: int = const 39;
    b: int = const 39;
    c: int = add a b;
}
```
When parsing the thrid line of code, LVN replaces the arguments (`a`, `b`) with the numbers they refer to (number 1). 
Since the canonical variable of the value numbered 1 is `a`, both arguments are replaced with `a`.
```
c: int = add a a;
```

## Numbering is Not Necessary

In LVN, the situations where we need to access information include:
- Querying the number of the value a variable points to
- Assigning a value to a number and checking if a value exists
- Querying the canonical variable of a value

Therefore, the information we need to store include:
- A mapping from variables to numbers
- A mapping from values to numbers
- A mapping from numbers to canonical variables

When implementing LVN, I find it difficult to maintain three data structures. 
Trying to simplify the logic, I realized that **queries for the number of a value are actually queries the canoncial variable of the value**.  
The point of numbering is to make variables with identical values all refer to the same thing. 
Besides, when transforming the code, variables are replaced with canoncial variables.
Because every value is assigned to only one canonical variable, canonical variables can be used to identify values like numbers.  
Thus, we only need to store a mapping from variables to canonical variables, and values to canonical variables.  
The overall algorithm looks like this:
```
var2canon: Variable -> Canonical variable
val2canon: Value -> Canonical variable

for instruction in instructions
    value = create_value(instruction)
    instruction = transform_instruction(instruction, value, var2canon, val2canon)
end
```
- `create_value` is a function that creates the specific type of value based on the instruction.
- `transform_instruction` is a function that eliminates unused variable while maintaining the canonical variable table.


## Problems and Implementations

Let’s consider a trivial case to lay down the bare bones of the function.

### Constant Elimination
```
@main {
    a: int = const 39;
    b: int = const 39;  # Eliminate this line
}
```
The target is to eliminate `b`.  
We first declare an abstract type `Value` and let different types of values inherit it.
```julia
abstract type Value end
```
A constant value is determined solely by the numerical value it contains:
```julia
struct Constant <: Value
    value::Int
end
```
for transforming instructions, we need to consider 2 situations:
- If the value is already known and linked to a canonical variable:
  - Replace the instruction with an instruction linking to the canonical variable (operation "id")
  - Map the current variable ("dest") to the canonical variable
- If the value is a new value, update the canonical variable tables.

In code, it would be like this:
```julia
function transforminstr(instr, value::Constant, var2canon, val2canon)
    if value in keys(val2canon)  # Value is known
        var = val2canon[value]

        # Replace the instruction
        instr["op"] = "id"
        instr["args"] = [var]
        delete!(instr, "value")

        # Update the table with the current variable
        var2canon[instr["dest"]] = var
    else
        # Update the tables
        var = instr["dest"]
        var2canon[var] = var
        val2canon[value] = var
    end
end
```

The main algorithm would be like this:
```julia
function canonicalizelocalvalue(instrs)
    var2canon = Dict()
    val2canon = Dict()

    for instr in instrs
        value = if instr["op"] == "const"
            Constant(instr["value"])
        end
        transforminstr(instr, value, var2canon, val2canon)
    end
end
```

### Math Operation Elimination

```
@main {
    a: int = const 39;
    b: int = add a a;
    c: int = add a a;  # Eliminate this line
}
```
A unqiue math operation is determined by its operation and two operands that are canonical variables.
```julia
struct MathOp <: Value
    op::String
    opr1::String
    opr2::String
end

function MathOp(instr, var2canon)
    opr1, opr2 = (var2canon[var] for var in instr["args"])
    MathOp(instr["op"], opr2, opr2)
end
```
Transforming mathop instructions is slightly different from constant instructions. 
Specifically, if the operation produces a new value, its operands need to be canonicalized.
```julia
function transforminstr(instr, value::MathOp, var2canon, val2canon)
    if value in keys(val2canon)
        # ... Identical to constant case, omitted
    else
        instr["args"] = [value.opr1, value.opr2]
        # ... Identical to constant case, omitted
    end
end
```
Here I am using a feature in Julia called **multiple dispatch**. 
Basically, Julia automatically selects the function to execute based on the type of `value`.  
This provides multiple benefits during development:
- By considering only one type of value, I can write specific code for one scenario, which improves logic.
- I can continuously add new value types while changing little code.

The main algorithm only needs to add a new condition:
```julia
value = if instr["op"] == "const"
    Constant(instr["value"])
elseif instr["op"] in ["add", "mul", "sub", "div"]
    MathOp(instr, var2canon)
end
```

### Identity Elimination
```
@main {
    a: int = const 39;
    b: int = const 39;
    c: int = id b;  # Replace b with a
}
```
The value of an identity instruction is simply a reference to its canonical variable.
```julia
struct Identity <: Value
    ref_var::String
end

function Identity(instr, var2canon)
    var = instr["args"][1]  # Get the referenced variable
    Identity(var2canon[var])
end
```
An identity instruction has a trivial case of transformation: canonicalize the variable used and update the variable table.
```julia
function transform_instr(instr, value::Identity, var2canon, val2canon)
    instr["args"][1] = value.ref_var
    var2canon[instr["dest"]] = value.ref_var
end
```

## Conclusion

This is an article introducing **local value numbering** and putting my own spin on it. 
In addition, I provided an implementation in Julia, which demonstrates the feature **multiple dispatch**.  
A few caveats:
- This is my first attempt in writing Julia. I am not sure not sure whether I picked a decent abstraction and used multiple dispatch well. 
  **Please do not hesitate to give me suggestions!**
- The implementation fails to consider edge cases such as reassignments, and has great room for improvement.

Finally, I would like to thank [Professor Sampson](https://twitter.com/samps) for offering such a great compiler course, and I look forward to the future lessons!
