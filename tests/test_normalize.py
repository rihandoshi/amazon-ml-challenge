"""Tests for the normalize module, especially address parsing.

Run with:
    python -m pytest tests/test_normalize.py -v
    or simply:
    python tests/test_normalize.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalize import normalize_address, normalize_name, _extract_city_improved, _clean_basic


def test_greensboro_nc_address():
    """The known problematic case: 'GREENSBORO, NC, 19 1/2 STARDUST TRAIL'
    should extract 'greensboro' as the city, not '19 1/2 stardust trail'.
    """
    result = normalize_address("GREENSBORO, NC, 19 1/2 STARDUST TRAIL", "US")
    assert result["city"] is not None, "City should not be None"
    assert "greensboro" in result["city"].lower(), \
        f"Expected 'greensboro' but got '{result['city']}'"
    assert result["state"] == "NC", f"Expected state 'NC' but got '{result['state']}'"


def test_standard_us_address():
    """Standard US format: '1795 Westchester Drive, High Point, NC'"""
    result = normalize_address("1795 Westchester Drive, High Point, NC", "US")
    assert result["city"] is not None, "City should not be None"
    assert "high point" in result["city"].lower(), \
        f"Expected 'high point' but got '{result['city']}'"
    assert result["state"] == "NC"
    assert result["house_no"] == "1795"


def test_us_address_with_zip():
    """US address with zip: '100 Main St, Springfield, IL 62701'"""
    result = normalize_address("100 Main St, Springfield, IL 62701", "US")
    assert result["zip"] == "62701"
    assert result["state"] == "IL"


def test_indian_address():
    """Indian address: 'G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh'"""
    result = normalize_address("G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh", "India")
    assert result["city"] is not None, "City should not be None"
    assert "bhopal" in result["city"].lower(), \
        f"Expected 'bhopal' but got '{result['city']}'"
    assert result["state"] == "madhya pradesh"


def test_indian_address_with_pin():
    """Indian address with PIN code"""
    result = normalize_address("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi, 110018", "India")
    assert result["zip"] == "110018"


def test_address_with_unit():
    """Address with unit: '2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD'"""
    result = normalize_address("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "US")
    assert result["state"] == "MD"
    assert result["house_no"] is not None


def test_empty_address():
    result = normalize_address("", "US")
    assert result["norm_addr"] == ""
    assert result["zip"] is None
    assert result["city"] is None


def test_generic_postal_code():
    """Non-US/India country should try generic 5-digit postal code."""
    result = normalize_address("123 Rue de la Paix, 75002, Paris", "France")
    assert result["zip"] == "75002"


def test_normalize_name_basic():
    result = normalize_name("The ABC Corp.")
    assert result["norm_name"] != ""
    assert "abc" in result["norm_name"]
    assert "corporation" in result["norm_name"]


def test_normalize_name_empty():
    result = normalize_name("")
    assert result["norm_name"] == ""
    assert result["tokens"] == []


def test_normalize_name_abbreviations():
    result = normalize_name("XYZ Inc LLC")
    assert "incorporated" in result["norm_name"]
    assert "llc" in result["norm_name"]


def test_city_extraction_city_before_state():
    """City is before state in comma-separated segments."""
    city = _extract_city_improved(
        "123 Main St, Springfield, Illinois", "IL", None, "US"
    )
    assert city is not None
    assert "springfield" in city.lower()


def test_city_extraction_state_in_middle():
    """State appears in the middle (unusual format)."""
    city = _extract_city_improved(
        "GREENSBORO, NC, 19 1/2 STARDUST TRAIL", "NC", None, "US"
    )
    assert city is not None
    assert "greensboro" in city.lower()


def test_city_extraction_no_state():
    """No recognized state — should still attempt city extraction."""
    city = _extract_city_improved(
        "123 Rue de la Paix, Paris, 75002", None, "75002", "France"
    )
    # Should extract something (Paris or similar)
    assert city is not None


if __name__ == "__main__":
    # Simple test runner
    test_functions = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {fn.__name__}: {e}")
            failed += 1

    print(f"\n  {passed} passed, {failed} failed out of {passed+failed}")
    if failed > 0:
        sys.exit(1)
