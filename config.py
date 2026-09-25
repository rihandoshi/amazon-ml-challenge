"""Shared config / constants for the entity resolution pipeline."""

# Legal-suffix / common-word normalization for business names.
# Longer keys first is NOT required here because we match whole tokens after splitting.
NAME_TOKEN_MAP = {
    "corp": "corporation", "corporation": "corporation",
    "inc": "incorporated", "incorporated": "incorporated",
    "ltd": "limited", "limited": "limited",
    "llc": "llc", "llp": "llp",
    "pvt": "private", "private": "private",
    "co": "company", "company": "company", "cos": "company",
    "assoc": "associates", "associates": "associates",
    "grp": "group", "group": "group",
    "intl": "international", "international": "international",
    "svcs": "services", "svc": "service", "services": "services", "service": "service",
    "mfg": "manufacturing", "manufacturing": "manufacturing",
    "bros": "brothers", "brothers": "brothers",
    "and": "and", "&": "and",
    "the": "",  # drop leading/stray articles
}

# Tokens that carry almost no discriminative signal for blocking (stopword-ish
# for business names). NOT dropped from features, only from blocking keys.
NAME_STOPWORDS = {
    "the", "and", "of", "&", "a", "an", "for", "llc", "llp", "inc", "incorporated",
    "corp", "corporation", "ltd", "limited", "pvt", "private", "co", "company",
    "group", "services", "service", "international", "intl",
}

ADDRESS_TOKEN_MAP = {
    "rd": "road", "road": "road",
    "st": "street", "str": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard",
    "dr": "drive", "drive": "drive",
    "ln": "lane", "lane": "lane",
    "ct": "court", "court": "court",
    "pl": "place", "place": "place",
    "sq": "square", "square": "square",
    "apt": "apartment", "apartment": "apartment",
    "bldg": "building", "building": "building",
    "flr": "floor", "floor": "floor",
    "unit": "unit",
    "hwy": "highway", "highway": "highway",
    "pkwy": "parkway", "parkway": "parkway",
    "no": "number", "number": "number",
    "po": "postoffice",
    "near": "near",
    "opp": "opposite", "opposite": "opposite",
}

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY",
}
US_STATE_ABBRS = set(US_STATES.values())

INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "delhi", "new delhi", "west delhi",
    "east delhi", "north delhi", "south delhi", "jammu and kashmir", "ladakh",
    "puducherry", "chandigarh",
}

MODEL_ID_EMBEDDING = "intfloat/multilingual-e5-small"  # MIT license, ~118M params, handles Devanagari+Latin

RANDOM_SEED = 42
