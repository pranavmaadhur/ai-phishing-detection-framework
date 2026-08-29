import argparse
import csv
import ipaddress
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl


# Words treated as suspicious if they appear anywhere in the URL.
SUSPICIOUS_WORDS = (
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "banking",
)

# Common URL shortening services.
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
    "mcaf.ee",
}

# Common multi-part public suffixes used to improve subdomain counting.
MULTIPART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk",
    "com.au", "net.au", "org.au",
    "co.in", "firm.in", "net.in", "org.in", "gen.in", "ind.in",
    "co.jp", "ne.jp", "or.jp",
    "com.br", "com.cn", "com.sg", "com.my",
    "co.nz", "co.za",
}

FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_underscore",
    "num_slash",
    "num_at_symbol",
    "num_digits",
    "has_ip_address",
    "has_https",
    "num_subdomains",
    "has_suspicious_words",
    "num_query_params",
    "is_shortened_url",
    "domain_has_digits",
]


def parse_url(url):
    """Clean and parse a URL, even if http:// or https:// is missing."""
    url = (url or "").strip()

    parsed = urlsplit(
        url if "://" in url else "//" + url
    )

    hostname = (parsed.hostname or "").lower().rstrip(".")

    return url, parsed, hostname


def is_ip(hostname):
    """Return 1 if the hostname is an IPv4/IPv6 address, else 0."""
    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def count_subdomains(hostname):
    """Count meaningful subdomains, ignoring a normal 'www' prefix."""
    if not hostname or is_ip(hostname):
        return 0

    labels = [part for part in hostname.split(".") if part]

    if len(labels) <= 2:
        return 0

    last_two = ".".join(labels[-2:])

    suffix_parts = 2 if last_two in MULTIPART_SUFFIXES else 1

    # One registrable-domain label + public suffix labels.
    registrable_parts = suffix_parts + 1

    if len(labels) > registrable_parts:
        subdomains = labels[:-registrable_parts]
    else:
        subdomains = []

    # Ignore conventional www.
    if subdomains and subdomains[0] == "www":
        subdomains = subdomains[1:]

    return len(subdomains)


def is_shortened(hostname):
    """Return 1 when the hostname belongs to a known URL shortener."""
    return int(
        any(
            hostname == shortener
            or hostname.endswith("." + shortener)
            for shortener in SHORTENERS
        )
    )


def extract_features(url):
    """
    Convert one URL into the exact 15 features used by the ML dataset.
    Returns a dictionary in FEATURE_COLUMNS order.
    """
    raw, parsed, hostname = parse_url(url)
    lower_url = raw.lower()

    return {
        "url_length": len(raw),
        "domain_length": len(hostname),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_underscore": raw.count("_"),
        "num_slash": raw.count("/"),
        "num_at_symbol": raw.count("@"),
        "num_digits": sum(char.isdigit() for char in raw),
        "has_ip_address": is_ip(hostname),
        "has_https": int(parsed.scheme.lower() == "https"),
        "num_subdomains": count_subdomains(hostname),
        "has_suspicious_words": int(
            any(word in lower_url for word in SUSPICIOUS_WORDS)
        ),
        "num_query_params": len(
            parse_qsl(parsed.query, keep_blank_values=True)
        ),
        "is_shortened_url": is_shortened(hostname),
        "domain_has_digits": int(
            any(char.isdigit() for char in hostname)
        ),
    }


def process_csv(input_csv, output_csv):
    """
    Read a CSV containing:
        url,label

    Write:
        url + 15 extracted features + label
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    rows_written = 0

    with input_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as source, output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as destination:

        reader = csv.DictReader(source)

        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")

        required = {"url", "label"}
        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "Input CSV must contain 'url' and 'label' columns. "
                f"Missing: {sorted(missing)}"
            )

        fieldnames = ["url"] + FEATURE_COLUMNS + ["label"]

        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in reader:
            # Basic cleaning: remove accidental leading/trailing spaces.
            url = (row.get("url") or "").strip()
            label = (row.get("label") or "").strip()

            # Skip completely empty URL rows.
            if not url:
                continue

            features = extract_features(url)

            output_row = {"url": url}
            output_row.update(features)
            output_row["label"] = label

            writer.writerow(output_row)
            rows_written += 1

    return rows_written


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean phishing URL data and extract the 15 lexical URL "
            "features used by the phishing-detection model."
        )
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        default="phishing_url_dataset_balanced.csv",
        help="Input CSV containing url,label",
    )

    parser.add_argument(
        "output_csv",
        nargs="?",
        default="phishing_url_dataset_balanced_with_url_features.csv",
        help="Output feature CSV",
    )

    args = parser.parse_args()

    count = process_csv(args.input_csv, args.output_csv)

    print(f"Done. Wrote {count:,} rows to: {args.output_csv}")
    print("Feature order:")
    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")


if __name__ == "__main__":
    main()
