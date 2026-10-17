# Answer for psf__requests-1142

Only read this after you've tried it yourself (see `task.md`).

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
