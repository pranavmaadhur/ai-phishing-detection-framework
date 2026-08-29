import ipaddress
from urllib.parse import urlsplit, parse_qsl

SUSPICIOUS_WORDS = (
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "banking"
)

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "buff.ly",
    "is.gd",
    "cutt.ly",
    "rebrand.ly",
    "tiny.cc",
    "shorturl.at",
    "rb.gy",
    "lnkd.in",
    "bit.do",
    "mcaf.ee"
}

MULTIPART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk",
    "com.au", "net.au", "org.au",
    "co.in", "firm.in", "net.in", "org.in",
    "gen.in", "ind.in",
    "co.jp", "ne.jp", "or.jp",
    "com.br", "com.cn", "com.sg", "com.my",
    "co.nz", "co.za"
}


def parse_url(url):
    url = (url or "").strip()

    # Allows URLs even if http:// or https:// is missing
    parsed = urlsplit(
        url if "://" in url else "//" + url
    )

    hostname = (parsed.hostname or "").lower().rstrip(".")

    return url, parsed, hostname


def is_ip(hostname):
    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def count_subdomains(hostname):
    if not hostname or is_ip(hostname):
        return 0

    labels = [x for x in hostname.split(".") if x]

    if len(labels) <= 2:
        return 0

    last_two = ".".join(labels[-2:])

    if last_two in MULTIPART_SUFFIXES:
        suffix_parts = 2
    else:
        suffix_parts = 1

    registrable_parts = suffix_parts + 1

    if len(labels) > registrable_parts:
        subdomains = labels[:-registrable_parts]
    else:
        subdomains = []

    # Ignore normal www prefix
    if subdomains and subdomains[0] == "www":
        subdomains = subdomains[1:]

    return len(subdomains)


def is_shortened(hostname):
    return int(
        any(
            hostname == domain or
            hostname.endswith("." + domain)
            for domain in SHORTENERS
        )
    )


def extract_features(url):
    raw, parsed, hostname = parse_url(url)

    lower_url = raw.lower()

    features = {
        "url_length":
            len(raw),

        "domain_length":
            len(hostname),

        "num_dots":
            raw.count("."),

        "num_hyphens":
            raw.count("-"),

        "num_underscore":
            raw.count("_"),

        "num_slash":
            raw.count("/"),

        "num_at_symbol":
            raw.count("@"),

        "num_digits":
            sum(c.isdigit() for c in raw),

        "has_ip_address":
            is_ip(hostname),

        "has_https":
            int(parsed.scheme.lower() == "https"),

        "num_subdomains":
            count_subdomains(hostname),

        "has_suspicious_words":
            int(
                any(
                    word in lower_url
                    for word in SUSPICIOUS_WORDS
                )
            ),

        "num_query_params":
            len(
                parse_qsl(
                    parsed.query,
                    keep_blank_values=True
                )
            ),

        "is_shortened_url":
            is_shortened(hostname),

        "domain_has_digits":
            int(
                any(
                    c.isdigit()
                    for c in hostname
                )
            )
    }

    return features