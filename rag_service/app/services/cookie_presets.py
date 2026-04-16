"""
Cookie presets for common cookie consent banner services.

These cookies are designed to suppress cookie consent banners when capturing
webpages to PDF. They represent common "accept all" or "dismiss" states.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional
from urllib.parse import urlparse


@dataclass
class Cookie:
    """Represents a browser cookie for Gotenberg's Chromium."""
    name: str
    value: str
    domain: str
    path: str = "/"
    secure: bool = True
    httpOnly: bool = False
    sameSite: str = "Lax"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "domain": self.domain,
            "path": self.path,
            "secure": self.secure,
            "httpOnly": self.httpOnly,
            "sameSite": self.sameSite,
        }


# Preset cookies for major cookie consent providers
# These represent "accept all" or "banner dismissed" states

COOKIEBOT_CONSENT = [
    Cookie(
        name="CookieConsent",
        value="{stamp:%27-1%27%2Cnecessary:true%2Cpreferences:true%2Cstatistics:true%2Cmarketing:true%2Cmethod:%27explicit%27%2Cver:1%2Cutc:1720000000000%2Cregion:%27US%27}",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

ONETRUST_CONSENT = [
    Cookie(
        name="OptanonAlertBoxClosed",
        value="2024-01-01T00:00:00.000Z",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="OptanonConsent",
        value="isGpcEnabled=0&datestamp=Tue+Jan+01+2024&version=6.33.0&geolocation=US&consentId=consent-id-123&interactionCount=1&isIABGlobal=false&hosts=&consent=",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

OSANO_CONSENT = [
    Cookie(
        name="osano_consentmanager",
        value="acceptAll",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

GOOGLE_CONSENT = [
    Cookie(
        name="CONSENT",
        value="YES+cb.2024-01-01-00.p0.en+FX+",
        domain=".google.com",
        path="/",
        secure=True,
        sameSite="None",
    ),
]

YOUTUBE_CONSENT = [
    Cookie(
        name="CONSENT",
        value="YES+cb.2024-01-01-00.p0.en+FX+",
        domain=".youtube.com",
        path="/",
        secure=True,
        sameSite="None",
    ),
]

# Generic consent cookies that work on many sites
GENERIC_CONSENT = [
    Cookie(
        name="cookie_consent",
        value="accepted",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="cc_cookie",
        value="{level:[%22necessary%22%2C%22functionality%22%2C%22tracking%22%2C%22targeting%22]}",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="__cookie_consent",
        value="1",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="cmapi_cookie_privacy",
        value="permit_1,2,3,4",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# IAB Europe Transparency and Consent Framework v2
TCF_V2_CONSENT = [
    Cookie(
        name="euconsent-v2",
        value="CPSG8sAPSQ8sABADBENBfEoABAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# Usercentrics (German CMP used by Jina AI and others)
# UC_UI is the global Usercentrics interface
# Uses localStorage primarily but cookies can help suppress banner
USERCENTRICS_CONSENT = [
    Cookie(
        name="uc_settings",
        value="eyJjb25zZW50RGF0ZSI6IjIwMjQtMDEtMDFUMDA6MDA6MDAuMDAwWiIsImNvbnRyb2xsZXJJZCI6InVjLWF1dG8tY29uc2VudC0xMjMiLCJpc0NvbnNlbnRSZXF1aXJlZCI6dHJ1ZSwiaXNGaXJzdFZpc2l0IjpmYWxzZSwiaXNWYWxpZENvbnNlbnQiOnRydWUsImxhc3RDb25zZW50RGF0ZSI6IjIwMjQtMDEtMDFUMDA6MDA6MDAuMDAwWiIsImNvbnNlbnRJZCI6IjEyMzQ1Njc4OSIsInZlcnNpb24iOiIzLjAuMCIsInJlZ2lvbiI6IlVTIn0=",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="uc_user_interaction",
        value="true",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="ucData",
        value="eyJjb25zZW50ZWQiOnRydWUsInNlcnZpY2VzIjpbImdvb2dsZS1hbmFseXRpY3MiLCJnb29nbGUtdGFnLW1hbmFnZXIiXX0=",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    # Additional Usercentrics cookies for banner suppression
    Cookie(
        name="uc_consent",
        value="true",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="uc_banner_closed",
        value="true",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# CookieYes (popular lightweight CMP)
COOKIEYES_CONSENT = [
    Cookie(
        name="cookieyes-consent",
        value="consent:yes,all:true",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="cky-consent",
        value="yes",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# Quantcast Choice (Quantcast CMP)
QUANTCAST_CONSENT = [
    Cookie(
        name="qc_consent",
        value="true",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="__qca",
        value="consent-given",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# Didomi (French CMP, popular in EU)
DIDOMI_CONSENT = [
    Cookie(
        name="didomi_token",
        value="eyJ1c2VyX2lkIjpudWxsLCJjb25zZW50X3R5cGUiOiJmdW5jdGlvbmFsIiwiZXhwaXJlc19pbiI6MzE1MzYwMDAwLCJ2ZXJzaW9uIjoxLCJwdXJwb3NlX2lkcyI6WyJhbmFseXRpY3MiLCJtYXJrZXRpbmciXSwiY3JlYXRlZF9hdCI6IjIwMjQtMDEtMDFUMDA6MDA6MDAuMDAwWiIsInVwZGF0ZWRfYXQiOiIyMDI0LTAxLTAxVDAwOjAwOjAwLjAwMFoiLCJ2ZW5kb3JzX2VuYWJsZWQiOnsiZ29vZ2xlLWFuYWx5dGljcyI6dHJ1ZX0sInB1cnBvc2VzX2Rpc2FibGVkIjpbXX0=",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="didomi_consent",
        value="true",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# TrustArc (enterprise CMP)
TRUSTARC_CONSENT = [
    Cookie(
        name="trustarc_consent",
        value="accepted",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="cmapi_cookie_privacy",
        value="permit_1,2,3,4,5",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# Sourcepoint (enterprise CMP used by major publishers)
SOURCEPOINT_CONSENT = [
    Cookie(
        name="sp_consent",
        value="true",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="consentUUID",
        value="consent-given-123",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="sp_cmp_consent",
        value="accepted",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# iubenda (popular for small businesses)
IUBENDA_CONSENT = [
    Cookie(
        name="iubenda-consent",
        value="true",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="_iub_cs-",
        value="%7B%22consent%22%3Atrue%2C%22timestamp%22%3A%222024-01-01%22%7D",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# Termly (legal compliance platform)
TERMLY_CONSENT = [
    Cookie(
        name="termly_consent",
        value="accepted",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="tl_consent",
        value="true",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# Piwik Pro / Matomo consent
PIWIK_CONSENT = [
    Cookie(
        name="mtm_consent",
        value="true",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
    Cookie(
        name="ppms_privacy_",
        value="consent_given",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# Moove (another popular EU CMP)
MOOVE_CONSENT = [
    Cookie(
        name="moove_gdpr_popup",
        value="accepted",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# Borlabs Cookie (WordPress popular)
BORLABS_CONSENT = [
    Cookie(
        name="borlabs-cookie",
        value="%7B%22consents%22%3A%7B%22essential%22%3A%5B%5D%2C%22statistics%22%3A%5B%5D%2C%22marketing%22%3A%5B%5D%2C%22externalMedia%22%3A%5B%5D%2C%22functionality%22%3A%5B%5D%7D%2C%22domain%22%3A%22.example.com%22%7D",
        domain="",
        path="/",
        secure=True,
        sameSite="Lax",
    ),
]

# Combine all presets for maximum coverage
ALL_PRESETS = (
    COOKIEBOT_CONSENT
    + ONETRUST_CONSENT
    + OSANO_CONSENT
    + USERCENTRICS_CONSENT
    + COOKIEYES_CONSENT
    + QUANTCAST_CONSENT
    + DIDOMI_CONSENT
    + TRUSTARC_CONSENT
    + SOURCEPOINT_CONSENT
    + IUBENDA_CONSENT
    + TERMLY_CONSENT
    + PIWIK_CONSENT
    + MOOVE_CONSENT
    + BORLABS_CONSENT
    + GOOGLE_CONSENT
    + YOUTUBE_CONSENT
    + GENERIC_CONSENT
    + TCF_V2_CONSENT
)


def get_consent_cookies_for_url(url: str) -> List[Cookie]:
    """
    Get appropriate consent cookies for a given URL.
    
    Returns a list of cookies with domains set to match the URL's domain.
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    
    if not domain:
        return []
    
    # Remove port if present
    if ":" in domain:
        domain = domain.split(":")[0]
    
    # Prepare base domain for cookies (include parent domain for broader matching)
    # e.g., www.example.com -> .example.com for broader cookie matching
    parts = domain.split(".")
    if len(parts) > 2:
        base_domain = "." + ".".join(parts[-2:])
    else:
        base_domain = "." + domain if not domain.startswith(".") else domain
    
    cookies = []
    
    # Add domain-specific cookies
    if "google.com" in domain or "googleapis.com" in domain:
        for c in GOOGLE_CONSENT:
            cookies.append(Cookie(**{**asdict(c), "domain": c.domain or base_domain}))
    
    if "youtube.com" in domain or "youtu.be" in domain:
        for c in YOUTUBE_CONSENT:
            cookies.append(Cookie(**{**asdict(c), "domain": c.domain or base_domain}))
    
    # Add generic consent cookies for all domains
    for c in ALL_PRESETS:
        # Use the cookie's domain if set, otherwise use the URL's base domain
        cookie_domain = c.domain if c.domain else base_domain
        cookies.append(Cookie(**{**asdict(c), "domain": cookie_domain}))
    
    return cookies


def get_all_preset_cookies() -> List[Cookie]:
    """Get all preset cookies for maximum banner suppression coverage."""
    return ALL_PRESETS


def cookies_to_json(cookies: List[Cookie]) -> str:
    """Convert a list of Cookie objects to JSON string for Gotenberg."""
    import json
    return json.dumps([c.to_dict() for c in cookies])
