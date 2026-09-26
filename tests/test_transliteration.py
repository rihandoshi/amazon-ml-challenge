"""Quick test for generic transliteration and unseen data handling."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalize import transliterate_generic, transliterate_devanagari, has_non_latin, _clean_basic

print("=== Generic Transliteration Tests ===")

# French accented
r = transliterate_generic("Société Générale")
print(f"  French: Société Générale -> {r}")
assert "Soci" in r or "soci" in r.lower(), f"Unexpected: {r}"

# German umlaut
r = transliterate_generic("München")
print(f"  German: München -> {r}")
assert "nchen" in r, f"Unexpected: {r}"

# Devanagari (should use specialized map)
r = transliterate_generic("कंपनी")
print(f"  Devanagari: कंपनी -> {r}")
assert len(r) > 0, "Devanagari transliteration empty"

# Arabic (will be mostly stripped but shouldn't crash)
r = transliterate_generic("شركة عربية")
print(f'  Arabic: شركة عربية -> "{r}" (expected: mostly empty, handled by embeddings)')

# Mixed Latin + accents
r = transliterate_generic("Café de Paris")
print(f"  Mixed: Café de Paris -> {r}")
assert "Caf" in r, f"Unexpected: {r}"

# _clean_basic should handle everything without crashing
r = _clean_basic("Société Générale & Co.")
print(f"  _clean_basic French: Société Générale & Co. -> {r}")

r = _clean_basic("شركة عربية للتجارة")
print(f"  _clean_basic Arabic: شركة عربية للتجارة -> \"{r}\"")

r = _clean_basic("東京商事株式会社")
print(f'  _clean_basic CJK: 東京商事株式会社 -> "{r}"')

r = _clean_basic("Компания Россия")
print(f'  _clean_basic Cyrillic: Компания Россия -> "{r}"')

print()
print("=== has_non_latin Tests ===")
assert has_non_latin("مرحبا") == True, "Arabic should be non-latin"
assert has_non_latin("Hello") == False, "English should be latin"
assert has_non_latin("कंपनी") == False, "Devanagari is excluded (handled separately)"
print("  All has_non_latin tests passed")
print()

# Test that normalize_dataframe doesn't crash with unknown countries
print("=== Unknown Country Handling ===")
import pandas as pd
from normalize import normalize_dataframe

df = pd.DataFrame({
    "entity_id": ["test_1", "test_2", "test_3"],
    "business_name": ["Société Générale", "ABC Corp", "東京商事"],
    "business_address": ["75001 Paris, France", "123 Main St, NY", "東京都千代田区"],
    "country": ["France", "US", "Japan"],
})
result = normalize_dataframe(df)
assert len(result) == 3, "Should have 3 rows"
assert "norm_name" in result.columns
assert "norm_addr" in result.columns
print(f"  France entity: name='{result.iloc[0]['norm_name']}', zip='{result.iloc[0]['zip']}'")
print(f"  US entity: name='{result.iloc[1]['norm_name']}', state='{result.iloc[1]['state']}'")
print(f"  Japan entity: name='{result.iloc[2]['norm_name']}', zip='{result.iloc[2]['zip']}'")
print("  PASS: Unknown countries handled gracefully")

print()
print("ALL TESTS PASSED ✓")
