---
title: "My Journey with Nand2Tetris - Part 1"
date: 2021-03-06T10:53:55+08:00
draft: false
---

This is part 1 of the series "My Journey with Nand2Tetris", covering things I 
wish I knew before starting.

## What is Nand2Tetris?

Nand2Tetris is an introductory course to computer engineering, covering almost 
every level of abstraction, from gate-level logic (Nand gates) to operating 
system-level software (Tetris).

## Motivation to Take the Course

Last year, I decided to brush up my basic computer science skills and searched 
for learning resources on computer architecture. After going through numerous 
recommendations, I landed on this course. The reason I took this course is the 
fact that it is **project-based**, meaning the goal is to _implement_ an item 
(e.g., An ALU) after being armed with _just enough_ knowledge to do so. As 
someone who prefers learning by doing, I am deeply attracted to this way of 
teaching. In addtion to reviewing computer architecture and operating system 
concepts, the course covers compilers, which I happen to be interested in.  
However, I acknowledge that this is in fact an ambitious introductory course, 
which aims for breadth at the cost of some depth. Therefore, I hope to learn 
just enough to be able to read the more advanced topics.

## Things I Wish I Knew before Starting

### Follow the course on Coursera.

I had trouble finding the resources on the course website and figured it out 
eventually.  

> ![Icon Explanation](/icon_explanation.png)  
> These icons are links to (from left to right) the instructions, the slides, 
> and the book chapter. Yes, it took me a shamefully long time to find out those 
> were links.

I would still recommend following the course on Coursera since watching videos 
is much faster than reading and the book for the second part of the course 
is not available for free.

### Read other people's solution to learn the syntax of the HDL.

To be clear, **this is not encouraging you to cheat**. The HDL guide is great, 
but I had a hard time finding the functionality I want, namely, the slicing
syntax. I end up finding it by reading other people's solution.

### The HDL syntax

I believe the HDL is decently flexible despite being educational. Still, 
there are some confusing parts I wish I had learned earlier.

#### Signal names with underscore is NOT allowed.
After wrestling with mysterious errors for a while, I believe that 
`snake_case` signal names are not allowed.

#### Slicing
**Only input and output signals of modules can be sliced.** In other words, 
all internal wires cannot be sliced. Suppose I have an internal 2-bit wire 
`tmp` that needs to be split between modules `Bar` and `Thud`. Ideally, I 
would write:
```vhdl
Foo(in=a, out=tmp);
Bar(in=tmp[0], out=barOut);
Thud(in=tmp[1], out=thudOut);
```
In practice, it is the **output** signal that should be split:
```vhdl
Foo(in=a, out[0]=barIn, out[1]=thudIn);
Bar(in=barin, out=barOut);
Thud(in=thudin, out=thudOut);
```

#### Test scripts
The syntax of a `.tst` file is as follows:
```bash
load HDL_FILE, output-file OUT_FILE, compare-to CMP_FILE, output-list OUT_LIST;
```
`OUT_LIST` contains the output format of each signal, separated by a space.
`s%R.L.C.R` indicates signal `s` is represented as `R` (decimal `D` or binary 
`B`), taking up `C` characters, with `L` spaces on the left, `R` spaces on the 
right.

### How to see the full error message when compiling the HDL code.
Due to the fixed window size of the hardware simulator, the error message is 
truncated most of the time. I simply googled my way through the problem, 
throwing all the error message I have at the search box. It was only almost 
finishing project 3 did I realize the full error message can be shown by 
executing this command in the command line: `./HardwareSimulator.sh foo.tst`.

### Don't over-engineer.

This is merely my suggestion. In my opinion, the point of implementation is to 
grasp the concept, and the fact that the projects are often simplified version 
of the real thing, overly-optimizing does not increase one's understanding to 
the topic. For example, it's not worth my time replacing clear abstractions of 
an ALU with obscure combinational logic in order to minimize the number of 
Nand gates. Besides, I believe in reality, an ALU is way more complicated and 
there are algorithms to deal with optimizations.

## Conclusion
These are the things I wish I knew before starting, and I hope that this could 
help those who are taking this course. Thank you for reading and feel free to 
contact me at Twitter @kimbochen.
