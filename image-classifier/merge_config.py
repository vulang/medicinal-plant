"""
Shared merge configuration for model training and serving.
"""

# Classes with low performance that should be merged by family, targeting the
# smallest class within each family group.
MERGE_BY_FAMILY_CLASSES = {
    "10",
    "11",
    "110",
    "112",  # Magnoliaceae
    "113",
    "114",
    "115",
    "116",
    "139",
    "141",
    "142",
    "157",
    "158",
    "177",
    "178",
    "174",  # Rutaceae
    "179", 
    "180",
    "197",
    "189",
    "48",
    "56",
    "57",
    "70",
    "71",
    "72",
    "73",
    "9",
    "51",  # Umbelliferae
    "52",
    "55",
    "170",
    "28",
    "85",
    "86",
    "87",
    "88",
    "89",
    "90",
    "122",  # Liliaceae
    "123",
    "124",
    "125",
    "128",  # Labiatae
    "129",  # Labiatae
    "133",  # Fagaceae
    "134",
    "135",
    "136",
    "137",  # Gentianaceae
    "138",
    "167",  # Bufonidae
    "168",
    "186",
    "187",
    "171",  # Rhamnaceae
    "119",
    "192",  # Oleaceae
    "193",
    "29",   # Aristolochiaceae
    "30",
    "24",   # Amaranthaceae
    "25",
    "53"    # Araliaceae
    "97"
}

# Low performance classes that should be merged by common name rather than family.
COMMON_NAME_MERGE_CLASSES = {
    "149",
    "150",
    "151",
    "153",
    "154",
    "6",    # Membranous milk vetch
    "202",  # Membranous milk vetch
    "16",   # Licorice
    "17",   # Licorice
}
