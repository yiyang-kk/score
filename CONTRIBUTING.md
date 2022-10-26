# TL; DR:
## Gitflow diagram
![Gitflow](info/gitflow_psw.svg)

- All features should be developed in braches
- Documentation will be generated automatically from master



## TL; DR:

### Apply KISS.

- KEEP IT SIMPLE, STUPID
- if the code looks complex, it is and you should refactor the code.
- [Important Programming  "Rules  of Thumb"](http://www.wou.edu/las/cs/csclasses/cs161/Lectures/rulesofthumb.html)

### Use IDE.

- It will make your life easier, in the end - e.g. for bughunting
- Recommended: [VS Code](https://code.visualstudio.com/).

### Clean notebooks before commiting.

- [Use `nbstripout`.](https://github.com/kynan/nbstripout) 
- Install using `conda install -c conda-forge nbstripout`
- Open terminal in folder with repository
- Run `nbstripout --install`.
- Now all notebooks will have their outputs cleaned.

### Follow PEP8 (or Google style guide)

- [PEP8](https://www.python.org/dev/peps/pep-0008/)
- [Google style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
- Your code will become easier to read and especially - to maintain
- There exist VS Code extension to check for the style

____
____
____

## WORKFLOW RECOMMENDATIONS
(longer version)

### - Use Github Flow for development
- why: 
    - measurable
    - can see the changes made in the code
    - simplifies code review
        - address feedback inline
        - easier discussion
    - overall: good practice
- https://guides.github.com/introduction/flow/
- related:
    - [Learn Git Branching - Interactive tutorial](https://learngitbranching.js.org/)

### - Prefer use of proper Integrated development environment (IDE) for development instead of notebooks
- Presentation [I don't like notebooks by Joel Grus - author of Data Science from Scratch (O'Reilly)](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/edit#slide=id.g362da58057_0_1)
- presentation explains why
- recommended: Visual Studio Code. I will refer to VS Code extensions in this document.
- install [VS Code Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- NEW! not in the presentation - Python extension now allows for notebook-like workflow, with the benefits of the proper IDE.


## CODING RECOMMENDATIONS

### - FOLLOW PEP8 CODING STANDARDS
- [Official docs](https://www.python.org/dev/peps/pep-0008/)
- PEP8 is Bible of Python programming. 
- It is easy to write unreadable code. Much harder to maintain it.
- PEP8 helps you to find the unreadable parts.
- To make it work flawlessly - install linter and autoformatter.
- [Linting Python in Visual Studio Code](https://code.visualstudio.com/docs/python/linting) - `autopep8` (for automatic code formatting) + `pep8` linting
- Related: 
    - [What does Pythonic mean?](https://stackoverflow.com/questions/25011078/what-does-pythonic-mean)
    - *PEP 20*. Run `import this` anywhere in Python and have a look at the Zen of Python.
    - Run `import antigravity`.

### - USE MEANINGFUL VARIABLE NAMES
- [Clean code 101 — Meaningful names and functions](https://medium.com/coding-skills/clean-code-101-meaningful-names-and-functions-bf450456d90c)
- Variable name should explain what does the variable contains.
- Prefer longer and more readable code over short, but unreadable.
- __This is especially important when prefixing/suffixing (e.g. `X_temp -> temporary`)__
- Similarly, it is preferable to have names which are not describing data structure (such as `df` for pandas.DataFrame)
- Rather describe, what is in given data frame (i.e. `transactions`, `clients`, ...)
- NOTE: 
    - in order to be consistent, when naming dataframes `df`, we should then name all variables by their type. (strings, ints, lists...)
    - however, there exist use cases, when naming by data structure is preferable - i.e. function expect multiple input structures, and we have to make switch for `list`, `dict`, and `dataframe`...
- It is also preferable to have explanatory names over mathematical notation, for improved readability.
- Common examples: `i -> counter`, `X -> features`, `y -> target`
- Overall, the target is to have self-explanatory code, which is readable, even without docstrings.


### - FILL DOCSTRINGS
- use [VS Code `autoDocstring` extension](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) and fill it up.
- every method, private or visible, every class, every function should have filled up docstring, 
  which says:
    - what is this function good for
    - what does it expect as input (including type)
    - what is the expected result (including type).
- DO NOT FORGET TO UPDATE DOCSTRING, WHEN CHANGING CODE. There is nothing more confusing than lying docstring.

### - USE IMPLICIT TRUTHY AND FALSY VALUES
- when checking existence, we should use implicit falsy values of empty objects.
- this makes code less wordy and more concise and clear.
- 0 is always `False`, and so is empty list `[]`, `None` values, empty dict `{}`, empty pandas.DataFrame or almost any other object, which is empty.
- therefore there is no need to assign False value for the object to make sure tat the object is empty.

### - DO NOT USE EVAL()
- [The Art of Defensive Programming](https://medium.com/web-engineering-vox/the-art-of-defensive-programming-6789a9743ed4)
- should be used very carefully (and if possible, not at all) 
- can be security problem, as arbitrary code can be put into the argument, and it will be executed afterwards.
- if used - input should be validated that it contains what we expect (i.e. string long at most 25 letters, only specific names, etc.)
- if needed - try to use `ast.literal_eval`


### - USE STRING FORMATTING
- [Python 3's f-Strings: An Improved String Formatting Syntax (Guide)](https://realpython.com/python-f-strings/)
- Python provides us with a lot of ways to format strings. Use `.format` or f-strings.
- The `+` operator can be used as well, however only in simple cases.

### - BUILD PATHS PROPERLY
- [Python 3 Quick Tip: The easy way to deal with file paths on Windows, Mac and Linux](https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f)
- use `pathlib`

### - USE LIST (AND DICT) COMPREHENSIONS WHENEVER POSSIBLE
- [List comprehension in Python](https://hackernoon.com/list-comprehension-in-python-8895a785550b)
- use list comprehensions whenever possible, but don't let them be too long

### - USE LOGGING (ADVANCED)
- [Logging in Python](https://realpython.com/python-logging/)
- instead of 'print' statements, create logs
- do not forget to 'kill' the logger when it is not needed (during development)

### - USE CONFIGS (ADVANCED)
- [4 Ways to manage the configuration in Python](https://hackernoon.com/4-ways-to-manage-the-configuration-in-python-4623049e841b)
- declare 'changeable' variables at one place - preferably using external configuration file and config parser.
