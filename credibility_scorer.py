# credibility_scorer.py - paper credibility scoring

import datetime

_TOP_VENUES = [
    # CS / ML
    "neurips",
    "nips",
    "icml",
    "iclr",
    "cvpr",
    "iccv",
    "eccv",
    "aaai",
    "ijcai",
    "acl",
    "emnlp",
    "naacl",
    "sigir",
    "kdd",
    "jmlr",
    "tmlr",
    "pami",
    # General Science (high impact)
    "nature",
    "science",
    "cell",
    "pnas",
    "physical review letters",
    "physical review",
    "ieee",
    "transactions",
    # Materials Science
    "advanced materials",
    "nano letters",
    "acs nano",
    "acta materialia",
    "journal of materials science",
    "nature materials",
    "small",
    # Chemistry / Biology
    "journal of the american chemical society",
    "jacs",
    "angewandte chemie",
    "chemical science",
    "nature chemistry",
    "nature biotechnology",
    # Physics / Engineering
    "applied physics letters",
    "journal of applied physics",
    "npj",
    "iscience",
    # Medicine
    "lancet",
    "nejm",
    "new england journal",
    "nature medicine",
    "jama",
    # Astrophysics & Space Science
    "astrophysical journal",
    "apj",
    "apjl",
    "apjs",
    "monthly notices",
    "mnras",
    "astronomy and astrophysics",
    "aap",
    "astronomical journal",
    "aj",
    "pasp",
    "aas",
    "icarus",
    "nature astronomy",
    "astroparticle physics",
    "journal of cosmology",
    "classical and quantum gravity",
    "physical review d",
    # Physics (beyond what is already listed)
    "physical review a",
    "physical review b",
    "physical review c",
    "physical review e",
    "physical review x",
    "prx quantum",
    "reviews of modern physics",
    "communications physics",
    "new journal of physics",
    "journal of physics",
    "europhysics letters",
    "annalen der physik",
    "optica",
    "optics express",
    "optics letters",
    "photonics",
    # Materials Science & Engineering
    "nature communications",
    "npj computational materials",
    "scripta materialia",
    "materials today",
    "corrosion science",
    "wear",
    "tribology",
    "soft matter",
    "langmuir",
    "acs applied materials",
    "chemistry of materials",
    # Chemistry
    "nature chemical engineering",
    "chemical communications",
    "chemical engineering journal",
    "green chemistry",
    "reaction chemistry",
    "rsc advances",
    # Biology & Medicine (supplement to existing)
    "nature methods",
    "plos biology",
    "plos one",
    "nucleic acids research",
    "bioinformatics",
    "bmc",
    # Earth & Environmental Science
    "geophysical research letters",
    "journal of geophysical research",
    "nature geoscience",
    "climate dynamics",
    "atmospheric chemistry",
    # Engineering & Robotics
    "robotics and automation",
    "iros",
    "icra",
    "automatica",
    "control engineering",
    "mechatronics",
]

_TOP_INSTITUTIONS = [
    # CS-focused
    "mit",
    "stanford",
    "cmu",
    "carnegie mellon",
    "google",
    "deepmind",
    "microsoft research",
    "berkeley",
    "meta ai",
    "openai",
    # Global research universities (broad disciplines)
    "oxford",
    "cambridge",
    "eth zurich",
    "epfl",
    "tsinghua",
    "pku",
    "peking university",
    "harvard",
    "yale",
    "princeton",
    "caltech",
    "max planck",
    "cnrs",
    "inria",
    "tokyo",
    "kyoto",
    # Materials/Physics-specific
    "argonne",
    "oak ridge",
    "nist",
    "fraunhofer",
    "helmholtz",
    "lawrence berkeley",
    "slac",
    # Medical/Bio
    "johns hopkins",
    "mayo clinic",
    "wellcome",
    # China
    "chinese academy of sciences",
    "cas",
    "fudan",
    "zhejiang university",
    "sjtu",
    "shanghai jiao tong",
    "nanjing university",
    "ustc",
    "university of science and technology of china",
    "sun yat-sen",
    "harbin institute",
    "hit",
    "beihang",
    # South Korea / Japan
    "kaist",
    "postech",
    "snu",
    "seoul national",
    "yonsei",
    "osaka university",
    "tohoku university",
    "nagoya university",
    # Europe (non-UK)
    "delft",
    "tu delft",
    "kth",
    "chalmers",
    "ecole polytechnique",
    "sorbonne",
    "rwth aachen",
    "tu munich",
    "technical university of munich",
    "ku leuven",
    "utrecht",
    "leiden",
    # India
    "iit",
    "indian institute of technology",
    "isc bangalore",
    "tifr",
    # Canada / Australia
    "toronto",
    "waterloo",
    "mcgill",
    "ubc",
    "melbourne",
    "monash",
    "anu",
    # Additional international research labs
    "riken",
    "aist",
    "csiro",
    "kaust",
]

_DOMAIN_CATEGORY_PATTERNS = [
    # (keyword_patterns, matching_arxiv_categories)
    (
        ["sparse", "dictionary learning", "compressed sensing", "sparse coding"],
        ["cs.cv", "cs.lg", "cs.ai", "eess.sp", "stat.ml", "eess.iv"],
    ),
    (
        [
            "machine learning",
            "deep learning",
            "neural network",
            "ai ",
            "llm",
            "transformer",
            "reinforcement learning",
        ],
        ["cs.lg", "cs.ai", "cs.ne", "stat.ml"],
    ),
    (
        ["computer vision", "image", "video", "detection", "segmentation"],
        ["cs.cv", "eess.iv", "cs.lg"],
    ),
    (
        ["nlp", "natural language", "text", "language model", "speech"],
        ["cs.cl", "cs.lg", "eess.as"],
    ),
    (
        ["robotics", "robot", "manipulation", "planning", "autonomous"],
        ["cs.ro", "eess.sy", "cs.ai"],
    ),
    (
        [
            "material",
            "coating",
            "surface",
            "polymer",
            "alloy",
            "nanostructure",
            "corrosion",
            "adhesion",
            "hydrophobic",
            "anti-ice",
            "anti-icing",
        ],
        ["cond-mat.mtrl-sci", "physics.app-ph", "cond-mat.soft"],
    ),
    (
        ["quantum", "qubit", "photon", "entanglement"],
        ["quant-ph", "cond-mat.mes-hall"],
    ),
    (
        [
            "astrophysics",
            "galaxy",
            "stellar",
            "cosmology",
            "dark matter",
            "spectroscopy",
            "redshift",
            "telescope",
            "gravitational wave",
            "pulsar",
            "supernova",
            "neutron star",
            "black hole",
            "exoplanet",
            "radio astronomy",
            "interferometry",
        ],
        [
            "astro-ph.ga",
            "astro-ph.co",
            "astro-ph.im",
            "astro-ph.he",
            "astro-ph.ep",
            "astro-ph.sr",
            "gr-qc",
            "hep-ph",
            "hep-th",
        ],
    ),
    (
        ["climate", "weather", "atmospheric", "ocean", "fluid", "aerodynamic"],
        ["physics.flu-dyn", "physics.ao-ph", "physics.geo-ph"],
    ),
    (
        [
            "biology",
            "protein",
            "gene",
            "cell",
            "molecular",
            "genomics",
            "drug discovery",
            "biochemistry",
        ],
        ["q-bio.bm", "q-bio.cb", "q-bio.mn", "cs.lg"],
    ),
    (
        ["chemistry", "reaction", "catalyst", "synthesis", "molecule"],
        ["physics.chem-ph", "cond-mat.soft", "q-bio.bm"],
    ),
    (
        ["signal processing", "radar", "sonar", "audio", "sensor"],
        ["eess.sp", "eess.as", "cs.it"],
    ),
]


def _match_domain_categories(target_domain: str) -> list[str]:
    """Match target domain string to relevant arXiv categories using keywords."""
    domain_lower = target_domain.lower()
    for patterns, categories in _DOMAIN_CATEGORY_PATTERNS:
        if any(kw in domain_lower for kw in patterns):
            return categories
    # Default fallback: broad science categories
    return ["cs.lg", "stat.ml", "physics.app-ph", "cond-mat.mtrl-sci", "q-bio"]


def score_paper_credibility(
    paper: dict,
    target_domain: str = "the target research domain",
) -> dict:
    """
    Score one paper for credibility (0-100) and attach score breakdown.
    """
    scored = dict(paper)
    breakdown = {}
    venue_detected = None

    # 1) Venue score (+40)
    venue_score = 0
    comment = str(paper.get("comment", "") or "").lower()
    journal_ref = str(paper.get("journal_ref", "") or "").lower()
    title = str(paper.get("title", "") or "").lower()
    abstract = str(paper.get("abstract", "") or "").lower()
    venue_text = f"{comment} {journal_ref}"

    for venue in _TOP_VENUES:
        if venue in venue_text:
            venue_score = 40
            venue_detected = venue.upper()
            break

    # Fallback: if no venue found in comment/journal_ref, check if
    # the paper mentions a known venue in its abstract or title.
    # This gives partial credit (20 pts) — it suggests affiliation
    # with quality work but isn't a confirmed publication.
    if venue_score == 0:
        secondary_text = f"{title} {abstract} {comment}"
        for venue in _TOP_VENUES:
            if venue in secondary_text:
                venue_score = 20
                venue_detected = f"{venue.upper()} (inferred)"
                break

    breakdown["venue"] = venue_score

    # 2) Institution score (+10)
    institution_score = 0
    authors_text = " ".join(str(a) for a in paper.get("authors", [])).lower()
    affiliations_text = str(paper.get("affiliations", "") or "").lower()
    institution_text = f"{authors_text} {affiliations_text} {comment}"

    for inst in _TOP_INSTITUTIONS:
        if inst in institution_text:
            institution_score = 10
            break

    breakdown["institution"] = institution_score

    # 3) arXiv category match score (+15)
    category_score = 0
    primary_category = str(paper.get("primary_category", "") or "").lower()
    categories = [str(c).lower() for c in paper.get("categories", [])]
    all_cats = [primary_category] + categories

    relevant_cats = _match_domain_categories(target_domain)
    for cat in all_cats:
        if cat in relevant_cats:
            category_score = 15
            break

    breakdown["category_match"] = category_score

    # 3b) Primary category specificity bonus (+10)
    # Rewards papers whose primary category closely matches the target domain.
    # This adds variance that helps distinguish between papers in the same batch.
    primary_cat_bonus = 0
    if primary_category and relevant_cats:
        # Exact primary category match to a domain-relevant category
        if primary_category in relevant_cats:
            primary_cat_bonus = 10
        else:
            # Partial match: same top-level group (e.g. astro-ph.* matches astro-ph.*)
            primary_group = primary_category.split(".")[0] if "." in primary_category else primary_category
            for rc in relevant_cats:
                rc_group = rc.split(".")[0] if "." in rc else rc
                if primary_group == rc_group:
                    primary_cat_bonus = 5
                    break

    breakdown["primary_category_bonus"] = primary_cat_bonus

    # 4) Recency score (+20)
    recency_score = 0
    pub_date_str = paper.get("published_date", "")
    if pub_date_str:
        try:
            pub_date = datetime.datetime.strptime(pub_date_str, "%Y-%m-%d")
            days_ago = (datetime.datetime.now() - pub_date).days
            if days_ago <= 30:
                recency_score = 20
            elif days_ago <= 90:
                recency_score = 10
            elif days_ago <= 180:
                recency_score = 5
        except ValueError:
            pass

    breakdown["recency"] = recency_score

    # 5) Abstract richness score (+15)
    # arXiv abstracts average 300-600 chars; 400 chars is a substantive abstract.
    abstract_score = 0
    if len(abstract) > 400:
        abstract_score = 15
    elif len(abstract) > 200:
        abstract_score = 8

    breakdown["abstract_length"] = abstract_score

    total = sum(breakdown.values())
    # Cap at 100
    total = min(total, 100)
    scored["credibility_score"] = total
    scored["credibility_breakdown"] = breakdown
    scored["venue_detected"] = venue_detected
    return scored
