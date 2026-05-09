import os
from pydoc import text
import re
import pickle
import urllib.parse
import warnings
import requests
from bs4 import BeautifulSoup
import pandas as pd

_taxodist_cache = {}


def clear_cache():
    """
    Clear the taxodist lineage cache

    Clears all cached lineages stored in the current R session. Useful when
    you suspect cached data is stale or want to force fresh retrieval.

    Returns
    -------
    None
        Invisibly returns None.
    """
    global _taxodist_cache
    _taxodist_cache.clear()
    return None


def save_cache(file):
    """
    Save the taxodist lineage cache to disk

    Serialises the current session cache to an `.rds` file so it can be
    restored in a future session with load_cache(). Useful for
    reproducibility and for avoiding repeated network requests.

    Parameters
    ----------
    file : str
        Path to the `.rds` file to write.

    Returns
    -------
    None
        Invisibly returns None.
    """
    with open(file, 'wb') as f:
        pickle.dump(_taxodist_cache, f)
    print(f"Cache saved to '{file}' ({len(_taxodist_cache)} entries).")
    return None


def load_cache(file):
    """
    Load a previously saved taxodist cache from disk

    Restores lineage data saved with save_cache() into the current session
    cache, avoiding network requests for taxa already retrieved in a previous
    session.

    Parameters
    ----------
    file : str
        Path to an `.rds` file created by save_cache().

    Returns
    -------
    None
        Invisibly returns None.
    """
    global _taxodist_cache
    if not os.path.exists(file):
        raise FileNotFoundError(f"Cache file not found: '{file}'")
    with open(file, 'rb') as f:
        data = pickle.load(f)
        _taxodist_cache.update(data)
    print(f"Cache loaded from '{file}' ({len(data)} entries).")
    return None


def get_taxonomicon_id(taxon, verbose=False):
    """
    Find the Taxonomicon ID for a taxon name

    Queries The Taxonomicon (taxonomy.nl) to retrieve the internal numeric
    identifier for a given taxon name. The search filters out non-biological
    entities such as astronomical objects that may share the same name.

    Parameters
    ----------
    taxon : str
        A character string giving the taxon name to search for.
        Typically a genus name (e.g., "Tyrannosaurus") but species and higher
        ranks are also supported.
    verbose : bool
        Logical. If True, prints status messages during retrieval.
        Default is False.

    Returns
    -------
    str or None
        A character string containing the Taxonomicon numeric ID, or None
        if the taxon is not found.

    Details
    -------
    The function queries the static search endpoint at
    taxonomicon.taxonomy.nl/TaxonList.aspx and parses the resulting HTML
    to extract the taxon ID from the hierarchy link. When multiple matches
    exist (e.g., a genus name shared with an astronomical object), biological
    entries are prioritised by filtering for entries annotated as dinosaur,
    reptile, archosaur, animal, plant, fungus, or bacterium.
    """
    cache_key = f"id_{taxon}"
    if cache_key in _taxodist_cache:
        if verbose:
            print(f"Using cached ID for {taxon}")
        return _taxodist_cache[cache_key]

    if verbose:
        print(f"Searching Taxonomicon for {taxon}...")

    safe_taxon = urllib.parse.quote(str(taxon))
    url = f"http://taxonomicon.taxonomy.nl/TaxonList.aspx?subject=Entity&by=ScientificName&search={safe_taxon}"
    headers = {"User-Agent": "taxodist R package/0.3"}

    try:
        res = requests.get(url, headers=headers, timeout=30)
        if res.status_code != 200:
            warnings.warn(
                "Cannot reach The Taxonomicon server.\n"
                "The website (taxonomy.nl) appears to be offline or unreachable.\n"
                "Please try again later."
            )
            return None
    except requests.exceptions.RequestException:
        warnings.warn(
            "Cannot reach The Taxonomicon server.\n"
            "The website (taxonomy.nl) appears to be offline or unreachable.\n"
            "Please try again later."
        )
        return None

    soup = BeautifulSoup(res.text, "lxml")
    rows = soup.find_all("tr")
    bio_ids =[]

    for row in rows:
        text = row.get_text(separator=" ", strip=True)
        if re.search(r"astronomical|planet|Minor planet|comet|asteroid", text, flags=re.IGNORECASE):
            continue

        links_nodes = row.find_all("a", href=re.compile(r"TaxonTree"))
        if not links_nodes:
            continue

        valid_links =[a for a in links_nodes if "Valid" in a.get("class", [])]
        if not valid_links:
            continue

        target_link = valid_links[0]
        href = target_link.get("href", "")
        
        match = re.search(r"id=([0-9]+)", href)
        if not match:
            continue
        
        id_val = match.group(1)

        text_entry = re.sub(r"\s+", " ", text).strip()
        text_entry = re.sub(r"^N\s*\|\s*T\s*\|\s*P\s*\|\s*R\s*\|\s*B\s*\|\s*L\s*", "", text_entry, count=1)

        candidate_lin = get_lineage_by_id(id_val, clean=True, verbose=False)
        if candidate_lin is None or "Biota" not in candidate_lin:
            continue

        bio_ids.append({"id": id_val, "text": text_entry})

    if not bio_ids:
        if verbose:
            print(f"{taxon} not found in Taxonomicon")
        return None

    unique_ids =[]
    seen = set()
    for b in bio_ids:
        if b["id"] not in seen:
            seen.add(b["id"])
            unique_ids.append(b["id"])

    unique_bio_ids = []
    for uid in unique_ids:
        matches =[b for b in bio_ids if b["id"] == uid]
        unique_bio_ids.append(matches[0])

    if len(unique_bio_ids) > 1:
        warn_msg =[
            f"Multiple valid biological entries found for '{taxon}'.",
            f"Using: {unique_bio_ids[0]['text']} (ID: {unique_bio_ids[0]['id']})",
            f"To use a different entry, pass its numeric ID directly, e.g. `get_lineage(\"{unique_bio_ids[1]['id']}\")`.",
            "Other available IDs:"
        ]
        for i in range(1, len(unique_bio_ids)):
            warn_msg.append(f"* ID {unique_bio_ids[i]['id']}: {unique_bio_ids[i]['text']}")
        warnings.warn("\n".join(warn_msg))

    final_id = unique_bio_ids[0]["id"]
    _taxodist_cache[cache_key] = final_id
    if verbose:
        print(f"Found {taxon} with ID {final_id}")
        
    return final_id


def get_lineage_by_id(taxon_id, clean=True, verbose=False):
    """
    Retrieve the full taxonomic lineage of a taxon

    Given a Taxonomicon numeric ID, retrieves and parses the complete
    hierarchical lineage from root (Natura) to the taxon itself. The lineage
    is returned as a character vector ordered from root to tip.

    Parameters
    ----------
    taxon_id : str or int
        A numeric or character string giving the Taxonomicon ID.
        Obtain this with get_taxonomicon_id().
    clean : bool
        Logical. If True (default), removes philosophical root nodes
        above Biota (i.e., Natura, actualia, Mundus, naturalia) and strips
        dagger and superscript markers from names.
    verbose : bool
        Logical. If True, prints status messages. Default False.

    Returns
    -------
    list or None
        A character vector of clade names from root to tip, or None if
        retrieval fails.

    Details
    -------
    Lineage data is sourced from The Taxonomicon, which is based on
    Systema Naturae 2000 (Brands, S.J., 1989 onwards). The depth of lineages
    in The Taxonomicon substantially exceeds that of other programmatic sources
    such as the Open Tree of Life, particularly for well-studied clades such
    as Dinosauria, where intermediate clades at the level of superfamilies,
    tribes, and named subclades are fully resolved.
    """
    if taxon_id is None or str(taxon_id).strip() == "" or not re.match(r"^[0-9]+$", str(taxon_id)):
        return None

    taxon_id = str(taxon_id)
    cache_key = f"lin_{taxon_id}"

    if cache_key in _taxodist_cache:
        if verbose:
            print(f"Using cached lineage for ID {taxon_id}")
        return _taxodist_cache[cache_key]

    url = f"http://taxonomicon.taxonomy.nl/TaxonTree.aspx?id={taxon_id}&src=0"
    headers = {"User-Agent": "taxodist R package/0.3"}

    try:
        res = requests.get(url, headers=headers, timeout=30)
        if res.status_code != 200:
            if verbose:
                print(f"Could not retrieve lineage for ID {taxon_id}")
            return None
    except requests.exceptions.RequestException:
        if verbose:
            print(f"Could not retrieve lineage for ID {taxon_id}")
        return None

    soup = BeautifulSoup(res.text, "lxml")
    links = soup.find_all("a", href=re.compile(r"TaxonTree"))

    hrefs_all =[a.get("href", "") for a in links]
    valid_links =[a for a, h in zip(links, hrefs_all) if re.search(r"id=[0-9]", h)]
    hrefs_all =[h for h in hrefs_all if re.search(r"id=[0-9]", h)]

    hrefs = [a.get("href", "") for a in valid_links]
    own_pattern = f"id={taxon_id}(&|$)"
    
    own_idx = [i for i, h in enumerate(hrefs) if re.search(own_pattern, h)]
    if own_idx:
        valid_links = valid_links[:max(own_idx) + 1]

    lineage = []

    for a in valid_links:
        text = a.get_text(separator=" ", strip=True)        
        text = re.sub(r"[\u2020\u1D40†]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^\[crown\]\s*(Clade|Grandorder|Order|Superorder|Infraorder|Suborder|Class|Superclass|Subclass|Infraclass|Family|Superfamily|Subfamily|Tribe|Subtribe|Kingdom|Subkingdom|Infrakingdom|Domain|Superkingdom|Phylum|Subphylum|Genus|Species)?\s*", "", text)
        text = re.sub(r"^(Clade |Kingdom |Phylum |Superphylum |Subphylum |Infraphylum |Class |Order |Suborder |Infraorder |Parvorder |Grandorder |Magnorder |Cohort |Subcohort |Legion |Family |Subfamily |Tribe |Subtribe |Genus |Species |Subkingdom |Infrakingdom |Superclass |Subclass |Infraclass |Superorder |Superfamily |Domain |Superkingdom |Grade |Subgrade |Supergrade )", "", text)
        text = re.sub(r"\s+et\s+al\..*$", "", text)
        text = re.sub(r"\s+[A-Z][a-z]+-[A-Z][a-z]+.*$", "", text)
        text = re.sub(r"\s+[A-Z][a-z]+\s*&.*$", "", text)
        text = re.sub(r"\s+[A-Z][a-z]+,\s*\d{4}.*$", "", text)
        text = re.sub(r"\s+[A-Z][a-z]+\s+\d{4}.*$", "", text)
        text = re.sub(r"\s+[A-Z][a-z]+,.*$", "", text)
        text = re.sub(r"\s+[A-Z][a-z]+\s*$", "", text)
        text = re.sub(r"\s+[A-Z]\.[A-Z]\..*$", "", text)
        text = re.sub(r"\s+von\b.*$", "", text)
        text = re.sub(r"\s+auct\..*$", "", text)
        text = re.sub(r"\s+de\s+[A-Z].*$", "", text)
        text = re.sub(r"[\u2020\u1D40]", "", text)
        text = re.sub(r"\s+\([A-Z][a-z].*$", "", text)
        text = re.sub(r"\s+\(\d{4}\).*$", "", text)
        text = re.sub(r"\s+\[.*$", "", text)
        text = re.sub(r"\s+[A-Z]\.$", "", text)
        text = re.sub(r"\s+\([a-z].*$", "", text)
        text = re.sub(r'\s+".*$', "", text)
        text = re.sub(r'^".*', "", text)
        text = re.sub(r"\s+\d{4}.*$", "", text)
        text = text.strip()
        lineage.append(text)

    bare_ranks = {
        "Go to", "Subphylum", "Infraphylum", "Superphylum", "Subfamily", "Suborder",
        "Infraorder", "Superclass", "Subclass", "Superfamily",
        "Subgenus", "Section", "Division", "Candidatus", "Parvphylum",
        "Branch", "Supercohort", "Infracohort", "Subdivision", "Subsection",
        "Grade", "[unranked]", "(Supercluster)", "(Region)",
        "[crown]", ""
    }

    lineage = [x for x in lineage if x not in bare_ranks]
    lineage =[x for x in lineage if x != "" and not re.match(r"^\s*$", x)]
    lineage = [x for x in lineage if not x.startswith('"')]
    lineage = [x for x in lineage if not x.startswith("Population")]
    
    lineage = list(dict.fromkeys(lineage))

    if clean:
        try:
            idx_biota = lineage.index("Biota")
            lineage = lineage[idx_biota:]
        except ValueError:
            pass

    if not lineage:
        return None

    _taxodist_cache[cache_key] = lineage
    return lineage

def get_lineage(taxon, clean=True, verbose=False):
    """
    Retrieve the full taxonomic lineage of a taxon by name

    A convenience wrapper that combines get_taxonomicon_id() and
    get_lineage_by_id() into a single call. Given a taxon name, returns
    its complete lineage from root to tip.

    Parameters
    ----------
    taxon : str
        A character string giving the taxon name.
    clean : bool
        Logical. If True (default), removes philosophical root nodes
        and cleans formatting markers.
    verbose : bool
        Logical. If True, prints progress messages. Default False.

    Returns
    -------
    list or None
        A character vector of clade names ordered from root to tip, or
        None if the taxon cannot be found.
    """
    taxon_str = str(taxon)
    is_id = bool(re.match(r"^[0-9]+$", taxon_str))

    if is_id:
        id_val = taxon_str
    else:
        id_val = get_taxonomicon_id(taxon_str, verbose=verbose)

    if id_val is None:
        return None

    lineage = get_lineage_by_id(id_val, clean=clean, verbose=verbose)
    if lineage is None:
        return None

    if not is_id:
        if not re.search(r"\s", taxon_str):
            lineage =[x for x in lineage if not re.search(r" ", x)]
            lineage =[x for x in lineage if not x.startswith("[")]
            matches = [i for i, x in enumerate(lineage) if x == taxon_str]
            if matches:
                target_idx = matches[-1]
                lineage = lineage[:target_idx + 1]
            else:
                lineage.append(taxon_str)
        else:
            lineage =[x for x in lineage if not re.search(r" ", x) or x == taxon_str]
            matches = [i for i, x in enumerate(lineage) if x == taxon_str]
            if matches:
                target_idx = matches[0]
                lineage = lineage[:target_idx + 1]
            else:
                lineage.append(taxon_str)

    if not lineage:
        return None

    return lineage


def taxo_search(taxon, verbose=False):
    """
    Search The Taxonomicon for a taxon name

    Queries The Taxonomicon database and returns a data frame of all available
    biological entries matching the search string. This is particularly useful
    for exploring homonyms, historical ranks, or taxonomic synonyms before
    computing distances.

    Parameters
    ----------
    taxon : str
        A character string giving the taxon name to search for.
    verbose : bool
        Logical. If True, prints status messages. Default False.

    Returns
    -------
    pandas.DataFrame or None
        A data frame with columns:
        - id: Character. The numeric Taxonomicon ID.
        - name: Character. The full taxon description, including rank and author.
        Returns None if no matches are found.
    """
    if verbose:
        print(f"Searching Taxonomicon for '{taxon}'...")

    safe_taxon = urllib.parse.quote(str(taxon))
    url = f"http://taxonomicon.taxonomy.nl/TaxonList.aspx?subject=Entity&by=ScientificName&search={safe_taxon}"
    headers = {"User-Agent": "taxodist R package/0.3"}

    try:  
        res = requests.get(url, headers=headers, timeout=30)
        if res.status_code != 200:
            if verbose:
                print("Could not reach Taxonomicon")
            return None
    except requests.exceptions.RequestException:
        if verbose:
            print("Could not reach Taxonomicon")
        return None

    soup = BeautifulSoup(res.text, "lxml")
    rows = soup.find_all("tr")

    results =[]
    for row in rows:
        text = row.get_text(separator=" ", strip=True)
        if re.search(r"astronomical|planet|Minor planet|comet|asteroid", text, flags=re.IGNORECASE):
            continue

        links = row.find_all("a", href=re.compile(r"TaxonTree"))
        if not links:
            continue

        valid_links =[a for a in links if "Valid" in a.get("class", [])]
        if not valid_links:
            continue

        target_link = valid_links[0]
        href = target_link.get("href", "")
        
        match = re.search(r"id=([0-9]+)", href)
        if not match:
            continue

        id_val = match.group(1)

        text_entry = re.sub(r"\s+", " ", text).strip()
        text_entry = re.sub(r"^N\s*\|\s*T\s*\|\s*P\s*\|\s*R\s*\|\s*B\s*\|\s*L\s*", "", text_entry, count=1)

        results.append({"id": id_val, "name": text_entry})

    if not results:
        if verbose:
            print("No matches found.")
        return None

    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    if verbose:
        print(f"Found {len(df)} entries.")
        
    return df