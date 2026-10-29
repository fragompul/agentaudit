# ruff: noqa
"""Look up every bib entry in Crossref (journals/proceedings/books) or the arXiv API; print the top match."""

import json, re, sys, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

s = open("latex/references.bib", encoding="utf8").read()


def get(u):
    return urllib.request.urlopen(
        urllib.request.Request(u, headers={"User-Agent": "agentaudit-refcheck (mailto:none)"}),
        timeout=30,
    ).read()


for e in re.split(r"\n(?=@)", s):
    k = re.match(r"@(\w+)\{([^,]+),", e)
    if not k:
        continue

    def f(n):
        m = re.search(n + r"\s*=\s*[{\"](.*?)[}\"],?\s*\n", e, re.S)
        return re.sub(r"[{}\$]", "", m.group(1).replace("\n", " ")) if m else ""

    title, auth, year = f("title"), f("author"), f("year")
    print(
        "==",
        k.group(2),
        "|",
        auth[:60],
        "|",
        title[:70],
        "|",
        year,
        "|",
        f("journal") or f("booktitle") or f("publisher"),
    )
    try:
        ax = re.search(r"arXiv:(\d{4}\.\d{4,5})", e)
        if ax:
            r = ET.fromstring(get("http://export.arxiv.org/api/query?id_list=" + ax.group(1)))
            n = "{http://www.w3.org/2005/Atom}"
            en = r.find(n + "entry")
            print(
                "   arXiv:",
                en.find(n + "title").text.strip().replace("\n", " ")[:90],
                "|",
                [a.find(n + "name").text for a in en.findall(n + "author")][:3],
                "|",
                en.find(n + "published").text[:10],
            )
        else:
            q = urllib.parse.quote(title)
            d = json.loads(
                get(
                    f"https://api.crossref.org/works?query.bibliographic={q}&rows=1&select=title,author,issued,container-title,DOI,publisher"
                )
            )["message"]["items"][0]
            print(
                "   Crossref:",
                d["title"][0][:90],
                "|",
                [a.get("family") for a in d.get("author", [])][:3],
                "|",
                d["issued"]["date-parts"][0][0],
                "|",
                (d.get("container-title") or [d.get("publisher")])[0],
                "|",
                d["DOI"],
            )
    except Exception as ex:
        print("   lookup failed:", ex)
