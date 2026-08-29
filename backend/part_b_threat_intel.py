"""
PART B - Live Threat Intelligence Check (Cybersecurity)
========================================================
Single importable function for the backend teammate to drop into the
existing /predict endpoint as an extra signal:

    from part_b_threat_intel import check_threat_intel
    result = check_threat_intel(url)
    # result = {"known_phishing": True/False, "source": "OpenPhish"}

--------------------------------------------------------------------------
WHY OPENPHISH INSTEAD OF PHISHTANK (the "your call" decision)
--------------------------------------------------------------------------
Both are legitimate public phishing feeds, but for a same-day hackathon:

  - OpenPhish's free feed (https://openphish.com/feed.txt) needs NO
    registration, NO API key, and NO signup wait - you can start calling
    it right now. It's a plain-text list of currently active phishing
    URLs, refreshed every few hours.
  - PhishTank's API works without a key too, but PhishTank *recommends*
    registering for a key (https://phishtank.org/api_register.php) to
    avoid aggressive rate limiting - approval isn't always instant, and
    hitting a rate limit mid-demo would be worse than not using it.

So this function uses OpenPhish as the primary source. If your team gets
a PhishTank key before the demo, swap the URL/parsing logic in
`_fetch_feed()` - the rest of the function (caching, matching,
check_threat_intel's return shape) doesn't need to change.

--------------------------------------------------------------------------
DESIGN NOTES
--------------------------------------------------------------------------
- The feed is a few thousand URLs and only updates every few hours, so
  this function caches it in memory for CACHE_TTL_SECONDS instead of
  re-downloading it on every /predict call (that would add real network
  latency to every request and could get you rate-limited).
- If the feed fetch fails for any reason (venue wifi drops, feed is
  down, timeout), this fails SAFE - it returns known_phishing: False
  rather than crashing the /predict endpoint. A missing threat-intel
  signal shouldn't take down the whole pipeline; the URL and text
  classifiers still run.
- Matching is done both on the exact URL and on the domain, since a
  reported phishing URL might have a slightly different path/query
  string than what a user actually visited, but the domain is still
  the same known-bad site.

Requires: pip install requests
"""

import time
from urllib.parse import urlparse
import requests

OPENPHISH_FEED_URL = "https://openphish.com/feed.txt"
CACHE_TTL_SECONDS = 3 * 60 * 60  # refetch at most every 3 hours
FETCH_TIMEOUT_SECONDS = 5

_cache = {
    "urls": set(),       # exact URLs from the feed
    "domains": set(),    # just the domains, for looser matching
    "fetched_at": 0.0,
}


def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(url if "://" in url else f"http://{url}").netloc
        return netloc.lower().split(":")[0]  # strip port if present
    except Exception:
        return ""


def _fetch_feed() -> None:
    """Downloads the OpenPhish feed and refreshes the in-memory cache."""
    resp = requests.get(OPENPHISH_FEED_URL, timeout=FETCH_TIMEOUT_SECONDS)
    resp.raise_for_status()

    urls = set()
    domains = set()
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        urls.add(line)
        d = _domain_of(line)
        if d:
            domains.add(d)

    _cache["urls"] = urls
    _cache["domains"] = domains
    _cache["fetched_at"] = time.time()


def _ensure_fresh_cache() -> bool:
    """Returns True if the cache is usable (fresh or successfully refreshed)."""
    is_stale = (time.time() - _cache["fetched_at"]) > CACHE_TTL_SECONDS
    if not is_stale and _cache["fetched_at"] > 0:
        return True
    try:
        _fetch_feed()
        return True
    except Exception:
        # Network hiccup - if we have an old cache, still use it rather
        # than treating everything as "unknown". If we have nothing
        # cached at all, we genuinely can't answer.
        return _cache["fetched_at"] > 0


def check_threat_intel(url: str) -> dict:
    """
    Checks whether `url` is a currently-reported phishing domain in the
    OpenPhish public feed.

    Returns: {"known_phishing": True/False, "source": "OpenPhish"}
    """
    if not url or not url.strip():
        return {"known_phishing": False, "source": "OpenPhish"}

    cache_ok = _ensure_fresh_cache()
    if not cache_ok:
        # Fail safe: no crash, just report "not known" and say why via source.
        return {"known_phishing": False, "source": "OpenPhish (feed unavailable)"}

    url = url.strip()
    if url in _cache["urls"]:
        return {"known_phishing": True, "source": "OpenPhish"}

    domain = _domain_of(url)
    if domain and domain in _cache["domains"]:
        return {"known_phishing": True, "source": "OpenPhish"}

    return {"known_phishing": False, "source": "OpenPhish"}


if __name__ == "__main__":
    # quick smoke test - replace with a URL from https://openphish.com/feed.txt
    # to see a True result (entries there rotate as sites get taken down).
    print(check_threat_intel("https://example.com"))
