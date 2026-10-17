# My attempt: psf__requests-1142

I did not find the fix myself. I reproduced the bug (a plain GET printed `{'Content-Length': '0'}`) in a `python:3.8` Docker container, then asked Claude to do the edit. It gave me the reference fix, which I applied with a small script (my first try failed because Windows saved the file with different line endings, and the script changed nothing). After the fix, GET and HEAD with no body print `{}`, POST with no body prints `'0'` and POST with `data='abc'` prints `'3'`. `git diff --stat`: 1 file changed, 2 insertions, 1 deletion.
