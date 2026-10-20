Build a small URL shortener service in this folder, using only the Python standard library (http.server and sqlite3).

What it must do:
- POST /shorten with a JSON body {"url": "..."} returns {"code": "...", "short": "http://127.0.0.1:8090/<code>"}; codes are 6 URL-safe characters; a URL that is not http(s) gets HTTP 400.
- GET /<code> redirects to the original URL with the redirect status that is correct for a link that may change later; find out which status that is and why (from a reliable source on the web), and write the answer with its source in README.md.
- GET /stats/<code> returns {"url": ..., "hits": N, "created": ISO-8601 time}; an unknown code gets 404 everywhere.
- Data lives in shortener.db next to the code and survives a restart.

Also:
- tests/ with pytest tests that start the server on a free port and cover every rule above, including the restart.
- README.md: how to run it, the API, and the redirect-status answer with its source.

Done when: `python3 -m pytest -q` passes, and the service is left running on port 8090 (it must still be up after you finish).
Use whatever tools, skills or helper agents fit the work.
