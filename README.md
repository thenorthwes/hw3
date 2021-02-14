# hw3

This application can be run by launching `main.py` -- in its current configuration it will extract
features from train and then run perceptron for 150 itearations on the small data set.

- Running > `python` Python 3.8.2 (default, Nov  4 2020, 21:23:28) [Clang 12.0.0 (clang-1200.0.32.28)] on darwin
- `python main.py` 

Necessary data is stored in `/files/**` at the top of the directory

## Output

Output is written to `/files/out/**` (Top of the directory as well)

- Training data output is in `output.txt`
- Dev data is in `output.txtdev`
- Test data is in `output.texttest`

Intermediate results can be written out as well for producing a learning curve. That can be turned on and off with the boolean flag `PRINT_INTERVALS`.
These results are based on the training data as perceptron cycles are executed.
- Currently it writes out to `output.txt{n}` -- where `n` is in intervals of 5

Named Entity Features can be turned on and off with `boolean` flags. Perceptron cycles can be controlled on line `159`

