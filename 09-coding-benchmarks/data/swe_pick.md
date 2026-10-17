# SWE-bench Verified pick: psf__requests-1142

Try it yourself first. The answer is at the very bottom, under a clear heading.

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

Reproduce by hand:

```
git clone https://github.com/psf/requests.git
cd requests
git checkout 22623bd8c265b78b161542663ee980738441c307
python -c "import requests; r = requests.Request('GET', 'http://example.com/').prepare(); print(r.headers)"
```

You should see `Content-Length` with value `'0'` in the printed headers (the bug). After a correct fix, the same command prints headers without Content-Length. Also check HEAD (same result wanted) and `requests.Request('POST', 'http://example.com/').prepare().headers` (should still show `Content-Length: '0'`). This old code needs an old Python (it is a 2013 codebase; Python 2.x or an early Python 3 may be needed to import it; I did not test which versions run it today, unconfirmed). Reading and editing the code works without running it.

Where to look: `requests/models.py`, method `prepare_content_length(self, body)` (about line 388 at the base commit). It is called from `prepare_body`. Hint: it receives `body` but the request method is available as `self.method`.

## What a beginner does

1. Read `prepare_content_length` and note where `'0'` is set.
2. Decide when `'0'` should be set (no body AND method is not GET/HEAD).
3. Edit the function, then run the one-liner above to check.

---
---
---

# ANSWER (do not read before trying)

Gold patch (non-test part), from the upstream PR https://github.com/psf/requests/pull/1142 (diff fetched from https://patch-diff.githubusercontent.com/raw/psf/requests/pull/1142.diff). The PR also edited AUTHORS.rst and added the test; those parts are left out here.

```diff
diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -386,13 +386,14 @@ def prepare_body(self, data, files):
         self.body = body
 
     def prepare_content_length(self, body):
-        self.headers['Content-Length'] = '0'
         if hasattr(body, 'seek') and hasattr(body, 'tell'):
             body.seek(0, 2)
             self.headers['Content-Length'] = str(body.tell())
             body.seek(0, 0)
         elif body is not None:
             self.headers['Content-Length'] = str(len(body))
+        elif self.method not in ('GET', 'HEAD'):
+            self.headers['Content-Length'] = '0'
```

The test added by the same PR (for reference):

```diff
+    def test_no_content_length(self):
+        get_req = requests.Request('GET', httpbin('get')).prepare()
+        self.assertTrue('Content-Length' not in get_req.headers)
+        head_req = requests.Request('HEAD', httpbin('head')).prepare()
+        self.assertTrue('Content-Length' not in head_req.headers)
```

Why it works, in plain English: the old function wrote `Content-Length: 0` first, unconditionally, and then overwrote it only if there was a body. So a bodyless GET kept the `0`. The fix removes that unconditional line. Now the header is set in exactly three cases: the body can seek/tell (a file-like object: use its size), the body is some other non-None object (use its length), or there is no body AND the method is not GET or HEAD (then `0`, because methods like POST/PUT/PATCH with no body are expected to say so explicitly). For a bodyless GET or HEAD none of the branches runs, so no Content-Length header is sent, which is what the issue asked for and what the test asserts.
