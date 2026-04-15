#!/usr/bin/env python3
"""
Test script for filter list functionality.
Tests filter list download, caching, and URL blocking.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from services.web_capture_service import (
    _download_filter_list,
    _load_filter_list,
    _should_block_with_filter_list,
    FILTER_LIST_CACHE_PATH
)

def test_filter_list_download():
    """Test downloading filter list."""
    print("Testing filter list download...")
    success = _download_filter_list()
    if success:
        print("✓ Filter list downloaded successfully")
        return True
    else:
        print("✗ Filter list download failed")
        return False

def test_filter_list_load():
    """Test loading filter list from cache."""
    print("\nTesting filter list loading...")
    rules = _load_filter_list()
    if rules:
        print(f"✓ Filter list loaded successfully with rules")
        return True
    else:
        print("✗ Filter list loading failed")
        return False

def test_url_blocking():
    """Test URL blocking with filter list."""
    print("\nTesting URL blocking...")

    # Test URLs that should be blocked (cookie banner related)
    test_urls = [
        "https://www.example.com/cookie-consent.js",
        "https://cdn.example.com/cookie-banner.css",
        "https://example.com/consent-manager.js",
    ]

    blocked_count = 0
    for url in test_urls:
        if _should_block_with_filter_list(url):
            print(f"  ✓ Blocked: {url}")
            blocked_count += 1
        else:
            print(f"  - Not blocked: {url}")

    print(f"\nBlocked {blocked_count}/{len(test_urls)} test URLs")
    return blocked_count > 0

def main():
    print("=" * 60)
    print("Filter List Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Download", test_filter_list_download()))
    results.append(("Load", test_filter_list_load()))
    results.append(("URL Blocking", test_url_blocking()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed"))
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
