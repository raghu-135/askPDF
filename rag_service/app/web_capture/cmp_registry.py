"""
CMP Registry - Cookie Management Platform detection and strategy mapping.

Provides heuristic-based detection of CMP providers from URLs and HTML content,
and maps them to appropriate consent strategies.
"""

import re
from enum import Enum
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse


class CMPType(Enum):
    """Supported Cookie Management Platform types."""
    ONETRUST = "onetrust"
    COOKIEBOT = "cookiebot"
    USERCENTRICS = "usercentrics"
    DIDOMI = "didomi"
    OSANO = "osano"
    COOKIEYES = "cookieyes"
    QUANTCAST = "quantcast"
    TRUSTARC = "trustarc"
    SOURCEPOINT = "sourcepoint"
    IUBENDA = "iubenda"
    TERMLY = "termly"
    PIWIK = "piwik"
    MOOVE = "moove"
    BORLABS = "borlabs"
    GOOGLE = "google"
    YOUTUBE = "youtube"
    GENERIC = "generic"
    NONE = None


# Domain patterns that indicate specific CMPs
DOMAIN_PATTERNS: Dict[CMPType, List[str]] = {
    CMPType.GOOGLE: ["google.com", "googleapis.com"],
    CMPType.YOUTUBE: ["youtube.com", "youtu.be"],
}

# HTML markers that indicate specific CMPs
# Format: (CMPType, list of (attribute_pattern, content_pattern))
HTML_MARKERS: List[tuple[CMPType, List[tuple[str, str]]]] = [
    (CMPType.ONETRUST, [
        ("id", r"onetrust-consent-sdk|onetrust-banner-sdk|ot-sdk-btn"),
        ("class", r"onetrust|ot-sdk-container"),
        ("data-domain-script", r"onetrust"),
    ]),
    (CMPType.COOKIEBOT, [
        ("id", r"CybotCookiebotDialog|CookiebotBanner|CookieConsent"),
        ("class", r"CybotCookie|cookiebot"),
        ("data-cbid", r""),
    ]),
    (CMPType.USERCENTRICS, [
        ("id", r"usercentrics-root|uc-banner"),
        ("class", r"usercentrics|uc-banner"),
        ("data-usercentrics", r""),
    ]),
    (CMPType.DIDOMI, [
        ("id", r"didomi-popup|didomi-banner"),
        ("class", r"didomi|didomi-popup"),
        ("data-didomi", r""),
    ]),
    (CMPType.OSANO, [
        ("id", r"osano-cm|osano-banner"),
        ("class", r"osano|osano-cm"),
        ("data-osano", r""),
    ]),
    (CMPType.COOKIEYES, [
        ("id", r"cookieyes|cky-banner"),
        ("class", r"cookieyes|cky-consent"),
    ]),
    (CMPType.QUANTCAST, [
        ("id", r"qc-consent|qc-cmp"),
        ("class", r"qc-cmp|quantcast"),
    ]),
    (CMPType.TRUSTARC, [
        ("id", r"trustarc|truste-banner"),
        ("class", r"trustarc|truste"),
    ]),
    (CMPType.SOURCEPOINT, [
        ("id", r"sp-message|sp-banner"),
        ("class", r"sp-message|sp-banner"),
    ]),
    (CMPType.IUBENDA, [
        ("id", r"iubenda-cs|iubenda-banner"),
        ("class", r"iubenda|iubenda-cs"),
    ]),
    (CMPType.TERMLY, [
        ("id", r"termly-banner|termly-consent"),
        ("class", r"termly|termly-consent"),
    ]),
    (CMPType.PIWIK, [
        ("id", r"ppms-cmp|matomo-consent"),
        ("class", r"ppms|matomo-consent"),
    ]),
    (CMPType.MOOVE, [
        ("id", r"moove_gdpr|moove-cookie"),
        ("class", r"moove|moove_gdpr"),
    ]),
    (CMPType.BORLABS, [
        ("id", r"borlabs-cookie|BorlabsCookieBox"),
        ("class", r"borlabs|borlabs-cookie"),
    ]),
]

# Provider-specific wait expressions for dynamic content
CMP_WAIT_EXPRESSIONS: Dict[CMPType, Optional[str]] = {
    CMPType.ONETRUST: "document.getElementById('onetrust-consent-sdk') === null || document.getElementById('onetrust-consent-sdk').style.display === 'none'",
    CMPType.COOKIEBOT: "document.getElementById('CybotCookiebotDialog') === null || document.getElementById('CybotCookiebotDialog').style.display === 'none'",
    CMPType.USERCENTRICS: "document.getElementById('usercentrics-root') === null || document.getElementById('usercentrics-root').style.display === 'none'",
    CMPType.DIDOMI: "document.querySelector('.didomi-popup-container') === null || document.querySelector('.didomi-popup-container').style.display === 'none'",
    CMPType.OSANO: "document.getElementById('osano-cm-manage-dialog') === null || document.getElementById('osano-cm-manage-dialog').style.display === 'none'",
    CMPType.COOKIEYES: "document.querySelector('.cky-consent-container') === null || document.querySelector('.cky-consent-container').style.display === 'none'",
}

# Provider-specific localStorage keys for banner suppression
CMP_LOCALSTORAGE_OVERRIDES: Dict[CMPType, Dict[str, str]] = {
    CMPType.USERCENTRICS: {
        "uc_settings": '{"isValidConsent":true,"isFirstVisit":false}',
        "uc_user_interaction": "true",
    },
    CMPType.ONETRUST: {
        "OptanonConsent": "true",
        "OptanonAlertBoxClosed": "true",
    },
    CMPType.COOKIEBOT: {
        "CookieConsent": '{"necessary":true,"marketing":true}',
    },
    CMPType.COOKIEYES: {
        "cookieyes-consent": "yes",
    },
    CMPType.OSANO: {
        "osano_consentmanager": "acceptAll",
    },
    CMPType.GENERIC: {
        "cookie_consent": "accepted",
    },
}

# Provider-specific targeted CSS selectors (vendor-specific)
CMP_CSS_SELECTORS: Dict[CMPType, List[str]] = {
    CMPType.USERCENTRICS: ["#usercentrics-root", ".uc-banner"],
    CMPType.ONETRUST: ["#onetrust-consent-sdk", "#onetrust-banner-sdk", ".ot-sdk-container"],
    CMPType.COOKIEBOT: ["#CybotCookiebotDialog", ".CybotCookie", ".cookiebot-banner"],
    CMPType.DIDOMI: [".didomi-popup-container", ".didomi-banner"],
    CMPType.OSANO: ["#osano-cm-manage-dialog", ".osano-cm"],
    CMPType.COOKIEYES: [".cookieyes-banner", ".cky-consent-container"],
    CMPType.QUANTCAST: [".qc-cmp", ".quantcast-banner"],
    CMPType.TRUSTARC: [".trustarc-banner", ".truste-banner"],
    CMPType.SOURCEPOINT: ["#sp-message", ".sp-banner"],
    CMPType.IUBENDA: [".iubenda-cs", ".iubenda-banner"],
    CMPType.TERMLY: [".termly-banner", ".termly-consent"],
    CMPType.PIWIK: [".ppms-cmp", ".matomo-consent"],
    CMPType.MOOVE: [".moove_gdpr", ".moove-cookie-info-bar"],
    CMPType.BORLABS: [".borlabs-cookie", "#BorlabsCookieBox"],
}

# Generic CSS selectors as fallback
GENERIC_CSS_SELECTORS = [
    '.cookie-banner',
    '.cookie-consent',
    '#gdpr-banner',
    '.cc-window',
    '.cc-banner',
]

# Attribute-based heuristic selectors (more precise)
HEURISTIC_CSS_SELECTORS = [
    # Cookie/consent related
    '[id*="cookie" i]:is([style*="fixed"], [style*="sticky"], [style*="z-index"])',
    '[class*="cookie" i]:is([style*="fixed"], [style*="sticky"], [style*="z-index"])',
    '[id*="consent" i]:is([style*="fixed"], [style*="sticky"], [style*="z-index"])',
    '[class*="consent" i]:is([style*="fixed"], [style*="sticky"], [style*="z-index"])',
    '[id*="gdpr" i]:is([style*="fixed"], [style*="sticky"], [style*="z-index"])',
    '[class*="gdpr" i]:is([style*="fixed"], [style*="sticky"], [style*="z-index"])',
    '[id*="privacy" i]:is([style*="fixed"], [style*="sticky"], [style*="z-index"])',
    '[class*="privacy" i]:is([style*="fixed"], [style*="sticky"], [style*="z-index"])',
    # Dialog modals with consent keywords
    '[role="dialog"][aria-modal="true"]:has-text(cookie, consent, privacy, gdpr)',
    '[data-testid*="cookie" i]',
    '[data-testid*="consent" i]',
]


def detect_cmp(url: str, html_content: Optional[str] = None) -> CMPType:
    """
    Detect the CMP provider from URL and/or HTML content.
    
    Args:
        url: The URL to check
        html_content: Optional HTML content to analyze
        
    Returns:
        The detected CMP type, or CMPType.NONE if not detected
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Check domain patterns first
    for cmp_type, patterns in DOMAIN_PATTERNS.items():
        for pattern in patterns:
            if pattern in domain:
                return cmp_type
    
    # If no HTML content provided, we can't detect from markers
    if not html_content:
        return CMPType.NONE
    
    # Check HTML markers
    html_lower = html_content.lower()
    
    for cmp_type, markers in HTML_MARKERS:
        for attr, pattern in markers:
            if pattern:
                # Look for the pattern in the attribute
                search_pattern = rf'{attr}=["\'][^"\']*{pattern}[^"\']*["\']'
                if re.search(search_pattern, html_content, re.IGNORECASE):
                    return cmp_type
            else:
                # Just check if attribute exists
                search_pattern = rf'{attr}=["\']'
                if re.search(search_pattern, html_content, re.IGNORECASE):
                    return cmp_type
    
    # Check for generic consent keywords if no specific CMP detected
    consent_indicators = [
        r'id=["\'][^"\']*cookie["\']',
        r'class=["\'][^"\']*cookie["\']',
        r'class=["\'][^"\']*consent["\']',
    ]
    for indicator in consent_indicators:
        if re.search(indicator, html_content, re.IGNORECASE):
            return CMPType.GENERIC
    
    return CMPType.NONE


def get_strategy_for_cmp(cmp_type: CMPType) -> Dict[str, Any]:
    """
    Get the consent strategy configuration for a given CMP type.
    
    Args:
        cmp_type: The detected CMP type
        
    Returns:
        Dictionary with strategy configuration:
        - localstorage_overrides: Dict of localStorage keys/values
        - css_selectors: List of CSS selectors to hide
        - wait_expression: Optional JavaScript expression to wait for
        - wait_delay: Recommended wait delay
    """
    if cmp_type == CMPType.NONE:
        return {
            "localstorage_overrides": {},
            "css_selectors": GENERIC_CSS_SELECTORS,
            "wait_expression": None,
            "wait_delay": "2s",
        }
    
    localstorage = CMP_LOCALSTORAGE_OVERRIDES.get(cmp_type, {}).copy()
    css_selectors = CMP_CSS_SELECTORS.get(cmp_type, GENERIC_CSS_SELECTORS).copy()
    wait_expression = CMP_WAIT_EXPRESSIONS.get(cmp_type)
    wait_delay = "2s" if cmp_type != CMPType.NONE else "2s"
    
    # Always include generic selectors as fallback
    if cmp_type not in CMP_CSS_SELECTORS:
        css_selectors.extend(GENERIC_CSS_SELECTORS)
    
    return {
        "localstorage_overrides": localstorage,
        "css_selectors": css_selectors,
        "wait_expression": wait_expression,
        "wait_delay": wait_delay,
    }


def get_fallback_strategy() -> Dict[str, Any]:
    """
    Get the fallback (broader) strategy for retry attempts.
    
    Returns:
        Dictionary with broader strategy configuration
    """
    return {
        "localstorage_overrides": {
            **CMP_LOCALSTORAGE_OVERRIDES.get(CMPType.GENERIC, {}),
            "cookie_consent": "accepted",
            "cc_cookie": '{"level":["necessary","functionality","tracking","targeting"]}',
        },
        "css_selectors": GENERIC_CSS_SELECTORS + [
            sel.replace(":is([style*=\"fixed\"], [style*=\"sticky\"], [style*=\"z-index\"])", "")
            for sel in HEURISTIC_CSS_SELECTORS
        ],
        "wait_expression": None,
        "wait_delay": "3s",
    }


def get_domain_from_url(url: str) -> str:
    """Extract base domain from URL for cookie domain setting."""
    parsed = urlparse(url)
    domain = parsed.netloc
    
    if not domain:
        return ""
    
    # Remove port if present
    if ":" in domain:
        domain = domain.split(":")[0]
    
    # Prepare base domain (include parent domain for broader matching)
    parts = domain.split(".")
    if len(parts) > 2:
        return "." + ".".join(parts[-2:])
    else:
        return "." + domain if not domain.startswith(".") else domain


def get_minimal_cookie_preset(cmp_type: CMPType) -> List[str]:
    """
    Get the minimal set of cookie names for a given CMP type.
    
    Args:
        cmp_type: The detected CMP type
        
    Returns:
        List of cookie preset names to use
    """
    from app.web_capture import cookie_presets
    
    cookie_map = {
        CMPType.ONETRUST: ["ONETRUST_CONSENT"],
        CMPType.COOKIEBOT: ["COOKIEBOT_CONSENT"],
        CMPType.USERCENTRICS: ["USERCENTRICS_CONSENT"],
        CMPType.DIDOMI: ["DIDOMI_CONSENT"],
        CMPType.OSANO: ["OSANO_CONSENT"],
        CMPType.COOKIEYES: ["COOKIEYES_CONSENT"],
        CMPType.QUANTCAST: ["QUANTCAST_CONSENT"],
        CMPType.TRUSTARC: ["TRUSTARC_CONSENT"],
        CMPType.SOURCEPOINT: ["SOURCEPOINT_CONSENT"],
        CMPType.IUBENDA: ["IUBENDA_CONSENT"],
        CMPType.TERMLY: ["TERMLY_CONSENT"],
        CMPType.PIWIK: ["PIWIK_CONSENT"],
        CMPType.MOOVE: ["MOOVE_CONSENT"],
        CMPType.BORLABS: ["BORLABS_CONSENT"],
        CMPType.GOOGLE: ["GOOGLE_CONSENT"],
        CMPType.YOUTUBE: ["YOUTUBE_CONSENT"],
        CMPType.GENERIC: ["GENERIC_CONSENT"],
        CMPType.NONE: [],
    }
    
    return cookie_map.get(cmp_type, ["GENERIC_CONSENT"])
