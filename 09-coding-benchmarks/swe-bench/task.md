# SWE-bench Verified task: psf__requests-1142

Try it yourself first. The real fix is in `answer.md`; don't open it until you've tried.

## Facts

- instance_id: `psf__requests-1142`
- repo: `psf/requests` (the Python HTTP library), library version 1.1
- base commit: `22623bd8c265b78b161542663ee980738441c307`
- issue: https://github.com/psf/requests/issues/1051 (title: "requests.get is ALWAYS sending content length", opened 2012-12-20 by orenbarzilai)
- upstream fix: PR https://github.com/psf/requests/pull/1142 (merged 2013-01-26). The instance number 1142 is the PR number, not the issue number.
- file changed (non-test): `requests/models.py`, method `PreparedRequest.prepare_content_length`
- size of the real fix: 1 line removed, 2 lines added in that one file

## How I checked it is in SWE-bench Verified

I queried the Hugging Face dataset-server search endpoint for the dataset `princeton-nlp/SWE-bench_Verified` (split `test`) with the text `psf__requests-1142`. It returned a row with instance_id `psf__requests-1142`, repo `psf/requests`, base_commit `22623bd8...` (as above), version 1.1 and the FAIL_TO_PASS below (row index 290 in that listing). The same query also returned psf__requests-2931 and psf__requests-1766, which fits the Verified requests subset. I could not download the whole dataset (the proxy blocks it), so I did not count the 500 rows. The `problem_statement` field was not retrievable verbatim, so the issue text below is from the public GitHub issue (see the note).

## The issue text (public GitHub text, issue #1051, original post)

Retrieved as separate quoted fragments (my fetch tool limits quote length, so I could not copy it in one piece). Line breaks and exact spacing may differ from GitHub. The author is orenbarzilai.

> Hi,
>
> It seems like that request.get always adds 'content-length' header to the request.
> I think that the right behavior is not to add this header automatically in GET requests or add the possibility to not send it.
> For example http://amazon.com returns 503 for every get request that contains 'content-length' header.
>
> Thanks,
> Oren

Note: sentence 3 came back split into two fragments ("...automatically in GET requests" and "or add the possibility to not send it."); I joined them. The SWE-bench `problem_statement` for this instance may also include some of the thread's comments; I could not read it verbatim. Gist of the thread (my paraphrase, not a quote): maintainers agreed Requests started adding the header to every request after an earlier change (#957); some servers such as amazon.com answered 503; the agreed fix was to stop adding it to GET (and HEAD).

## Tests that must go from failing to passing (FAIL_TO_PASS)

- `test_requests.py::RequestsTestCase::test_no_content_length`

What it checks (this is the upstream test added in the PR, so the test file at the base commit does not yet have it; SWE-bench adds it when evaluating): it builds a prepared GET request and a prepared HEAD request with `requests.Request(...).prepare()` and asserts that `'Content-Length'` is NOT in `.headers` for either one. It does not need the network, because it only prepares the requests and never sends them.

## The failing behaviour, in plain language

At the base commit, every request, even a GET with no body, gets the header `Content-Length: 0`. The goal is: GET and HEAD requests with no body should send no Content-Length header at all. Other methods with no body (for example POST with no data) should still send `Content-Length: 0`, and any request with a body should still send the real length.

Where to look: `requests/models.py`, method `prepare_content_length(self, body)` (about line 388). It is called from `prepare_body`. It receives `body`, and the request method is available as `self.method`.

## Reproduce it and try your fix

This 2013 code only runs on an old Python (3.9 or older), so run it in a throwaway Docker container with Python 3.8. The code stays on your PC; the container only borrows the folder. Run these in PowerShell:

```
cd C:\Users\brian
git clone https://github.com/psf/requests.git swe-requests
cd swe-requests
git checkout 22623bd8c265b78b161542663ee980738441c307
```

Show the bug (you should see `'Content-Length': '0'` for a plain GET):

```
docker run --rm -v ${PWD}:/r -w /r python:3.8 python -W ignore -c "import requests as r; print(r.Request('GET','http://example.com/').prepare().headers)"
```

Then open `requests\models.py` in VS Code, change `prepare_content_length`, and run the command again. Goal: GET and HEAD without a body print `{}`; POST without a body still prints `'Content-Length': '0'`; POST with `data='abc'` prints `'3'`. You can check all four at once:

```
docker run --rm -v ${PWD}:/r -w /r python:3.8 python -W ignore -c "import requests as r; [print(m, kw, r.Request(m,'http://e.com/',**kw).prepare().headers) for m,kw in [('GET',{}),('HEAD',{}),('POST',{}),('POST',{'data':'abc'})]]"
```

Write down what you tried (even the wrong tries) in `my_attempt.md`, then read `answer.md`.

Undo your edit at any time with `git checkout .`
