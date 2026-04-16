"""
Capture Strategy - Modular classes for webpage-to-PDF conversion.

Provides strategy classes for:
- Consent handling (cookies, localStorage, CSS suppression)
- Rendering (Gotenberg options, adaptive wait logic)
- Validation (banner detection with confidence scoring)
"""

import json
import base64
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from app.web_capture.cmp_registry import (
    CMPType,
    get_strategy_for_cmp,
    get_fallback_strategy,
    get_minimal_cookie_preset,
    get_domain_from_url,
)
from app.web_capture.cookie_presets import (
    Cookie,
    cookies_to_json,
    ONETRUST_CONSENT,
    COOKIEBOT_CONSENT,
    USERCENTRICS_CONSENT,
    DIDOMI_CONSENT,
    OSANO_CONSENT,
    COOKIEYES_CONSENT,
    QUANTCAST_CONSENT,
    TRUSTARC_CONSENT,
    SOURCEPOINT_CONSENT,
    IUBENDA_CONSENT,
    TERMLY_CONSENT,
    PIWIK_CONSENT,
    MOOVE_CONSENT,
    BORLABS_CONSENT,
    GOOGLE_CONSENT,
    YOUTUBE_CONSENT,
    GENERIC_CONSENT,
)


@dataclass
class CaptureOptions:
    """Configuration object for webpage capture parameters."""
    paper_size: str = "A4"  # A4, Letter, Legal
    wait_delay: str = "2s"
    wait_for_expression: Optional[str] = None
    wait_for_selector: Optional[str] = None
    user_agent: Optional[str] = None  # Default: desktop
    skip_network_almost_idle: bool = True
    max_retries: int = 1
    
    # Paper size dimensions in inches
    PAPER_SIZES = {
        "A4": (8.27, 11.69),
        "Letter": (8.5, 11),
        "Legal": (8.5, 14),
    }
    
    def get_paper_dimensions(self) -> Tuple[float, float]:
        """Get paper width and height in inches."""
        return self.PAPER_SIZES.get(self.paper_size, self.PAPER_SIZES["A4"])


class ConsentStrategy:
    """Strategy for handling cookie consent banner suppression."""
    
    @staticmethod
    def get_cookies(url: str, cmp_type: CMPType, fallback: bool = False) -> List[Cookie]:
        """
        Get the minimal set of cookies for the detected CMP type.
        
        Args:
            url: The URL being captured
            cmp_type: The detected CMP type
            fallback: If True, use broader cookie set for retry attempts
            
        Returns:
            List of Cookie objects
        """
        domain = get_domain_from_url(url)
        cookies = []
        
        # Get preset names based on CMP type
        if fallback:
            # Broader set for retry: include GENERIC_CONSENT
            preset_names = get_minimal_cookie_preset(cmp_type)
            if "GENERIC_CONSENT" not in preset_names:
                preset_names.append("GENERIC_CONSENT")
        else:
            preset_names = get_minimal_cookie_preset(cmp_type)
        
        # Map preset names to actual cookie lists
        preset_map = {
            "ONETRUST_CONSENT": ONETRUST_CONSENT,
            "COOKIEBOT_CONSENT": COOKIEBOT_CONSENT,
            "USERCENTRICS_CONSENT": USERCENTRICS_CONSENT,
            "DIDOMI_CONSENT": DIDOMI_CONSENT,
            "OSANO_CONSENT": OSANO_CONSENT,
            "COOKIEYES_CONSENT": COOKIEYES_CONSENT,
            "QUANTCAST_CONSENT": QUANTCAST_CONSENT,
            "TRUSTARC_CONSENT": TRUSTARC_CONSENT,
            "SOURCEPOINT_CONSENT": SOURCEPOINT_CONSENT,
            "IUBENDA_CONSENT": IUBENDA_CONSENT,
            "TERMLY_CONSENT": TERMLY_CONSENT,
            "PIWIK_CONSENT": PIWIK_CONSENT,
            "MOOVE_CONSENT": MOOVE_CONSENT,
            "BORLABS_CONSENT": BORLABS_CONSENT,
            "GOOGLE_CONSENT": GOOGLE_CONSENT,
            "YOUTUBE_CONSENT": YOUTUBE_CONSENT,
            "GENERIC_CONSENT": GENERIC_CONSENT,
        }
        
        # Build cookie list with proper domain assignment
        for preset_name in preset_names:
            preset_cookies = preset_map.get(preset_name, [])
            for cookie in preset_cookies:
                # Use the cookie's domain if set, otherwise use the URL's base domain
                cookie_domain = cookie.domain if cookie.domain else domain
                cookies.append(Cookie(
                    name=cookie.name,
                    value=cookie.value,
                    domain=cookie_domain,
                    path=cookie.path,
                    secure=cookie.secure,
                    httpOnly=cookie.httpOnly,
                    sameSite=cookie.sameSite,
                ))
        
        return cookies
    
    @staticmethod
    def get_localstorage_overrides(cmp_type: CMPType, fallback: bool = False) -> Dict[str, str]:
        """
        Get provider-specific localStorage key-value pairs.
        
        Args:
            cmp_type: The detected CMP type
            fallback: If True, include fallback localStorage overrides
            
        Returns:
            Dictionary of localStorage keys and values
        """
        strategy = get_strategy_for_cmp(cmp_type)
        overrides = strategy.get("localstorage_overrides", {}).copy()
        
        if fallback:
            fallback_strategy = get_fallback_strategy()
            overrides.update(fallback_strategy.get("localstorage_overrides", {}))
        
        return overrides
    
    @staticmethod
    def get_suppression_css(cmp_type: CMPType, fallback: bool = False) -> List[str]:
        """
        Get targeted CSS selectors for banner suppression.
        
        Args:
            cmp_type: The detected CMP type
            fallback: If True, use broader CSS selectors for retry
            
        Returns:
            List of CSS selectors
        """
        if fallback:
            fallback_strategy = get_fallback_strategy()
            return fallback_strategy.get("css_selectors", [])
        
        strategy = get_strategy_for_cmp(cmp_type)
        return strategy.get("css_selectors", [])
    
    @staticmethod
    def get_injection_script(cmp_type: CMPType, fallback: bool = False) -> str:
        """
        Generate immediate execution JavaScript for banner suppression.
        
        This script runs immediately (not waiting for DOMContentLoaded) to suppress
        banners before they can render.
        
        Args:
            cmp_type: The detected CMP type
            fallback: If True, use broader suppression script for retry
            
        Returns:
            JavaScript code as string
        """
        localstorage = ConsentStrategy.get_localstorage_overrides(cmp_type, fallback)
        css_selectors = ConsentStrategy.get_suppression_css(cmp_type, fallback)
        
        # Build localStorage setters
        localstorage_js = ""
        for key, value in localstorage.items():
            localstorage_js += f"try{{localStorage.setItem('{key}','{value}');}}catch(e){{}}"
        
        # Build CSS injection
        css_js = ""
        if css_selectors:
            selectors = ", ".join(css_selectors)
            css_js = f"var e=document.createElement('style');e.textContent='{selectors}{{display:none!important;visibility:hidden!important;opacity:0!important}}';document.head.appendChild(e);"
        
        # Combine into immediate-execution script (no DOMContentLoaded wait)
        script = f"{localstorage_js}{css_js}"
        
        return script


class RenderStrategy:
    """Strategy for building Gotenberg rendering options."""
    
    @staticmethod
    def build_gotenberg_options(
        capture_options: CaptureOptions,
        cmp_type: CMPType,
        cookies: List[Cookie],
        injection_script: str,
        fallback: bool = False,
    ) -> Dict[str, Tuple[Optional[str], str]]:
        """
        Build Gotenberg form fields for PDF conversion.
        
        Args:
            capture_options: Capture configuration
            cmp_type: The detected CMP type
            cookies: List of Cookie objects to send
            injection_script: JavaScript injection script
            fallback: Whether this is a retry attempt
            
        Returns:
            Dictionary of form fields for Gotenberg request
        """
        strategy = get_strategy_for_cmp(cmp_type)
        
        # Get paper dimensions
        paper_width, paper_height = capture_options.get_paper_dimensions()
        
        # Determine wait strategy: expression-based if available, otherwise delay
        wait_expression = capture_options.wait_for_expression
        wait_delay = capture_options.wait_delay
        
        if not wait_expression and cmp_type != CMPType.NONE:
            # Use CMP-specific wait expression if available
            cmp_wait = strategy.get("wait_expression")
            if cmp_wait:
                wait_expression = cmp_wait
            else:
                # Use strategy-recommended wait delay
                wait_delay = strategy.get("wait_delay", wait_delay)
        
        # If fallback (retry), use broader wait delay
        if fallback:
            fallback_strategy = get_fallback_strategy()
            wait_delay = fallback_strategy.get("wait_delay", "3s")
            wait_expression = None  # Fallback uses delay, not expression
        
        # Convert cookies to JSON
        cookies_json = cookies_to_json(cookies)
        
        # Build injection script (base64 encoded for data URL)
        script_b64 = base64.b64encode(injection_script.encode()).decode()
        extra_scripts = [{"src": f"data:text/javascript;base64,{script_b64}"}]
        
        # Build form fields
        form_fields: Dict[str, Tuple[Optional[str], str]] = {
            "paperWidth": (None, str(paper_width)),
            "paperHeight": (None, str(paper_height)),
            "marginTop": (None, "0"),
            "marginBottom": (None, "0"),
            "marginLeft": (None, "0"),
            "marginRight": (None, "0"),
            "printBackground": (None, "true"),
            "preferCssPageSize": (None, "false"),
            "cookies": (None, cookies_json),
            "extraScriptTags": (None, json.dumps(extra_scripts)),
            "emulatedMediaType": (None, "screen"),
        }
        
        # Add wait configuration
        if wait_expression:
            form_fields["waitForExpression"] = (None, wait_expression)
        else:
            form_fields["waitDelay"] = (None, wait_delay)
        
        # Add optional user agent
        if capture_options.user_agent:
            form_fields["userAgent"] = (None, capture_options.user_agent)
        
        return form_fields


class SuppressionValidator:
    """Validator for detecting remaining cookie banners after capture."""
    
    # Size threshold for considering an overlay a banner (200x200px)
    SIZE_THRESHOLD = 200 * 200
    
    # Banner detection patterns
    BANNER_INDICATORS = [
        # Position-based indicators
        (r'style=["\'][^"\']*position:\s*(fixed|sticky)', 0.3),
        (r'style=["\'][^"\']*z-index:\s*(\d{4,}|[1-9]\d{3})', 0.3),
        # ID/class based indicators
        (r'id=["\'][^"\']*(cookie|consent|gdpr|privacy|banner)', 0.5),
        (r'class=["\'][^"\']*(cookie|consent|gdpr|privacy|banner)', 0.5),
        # ARIA/dialog indicators
        (r'role=["\']dialog["\'].*aria-modal=["\']true["\']', 0.4),
        (r'data-testid=["\'][^"\']*(cookie|consent)', 0.6),
    ]
    
    @staticmethod
    def validate_banner_suppression(html_content: str) -> Tuple[bool, float]:
        """
        Validate whether cookie banners have been successfully suppressed.
        
        Uses DOM heuristics to detect visible overlays that match banner patterns.
        
        Args:
            html_content: HTML content from the captured page
            
        Returns:
            Tuple of (is_banner_present, confidence_score)
            - is_banner_present: True if a banner is likely present
            - confidence_score: 0.0-1.0 score of confidence
        """
        if not html_content:
            return False, 0.0
        
        # Look for potential banner elements
        banner_score = 0.0
        banner_count = 0
        
        # Search for suspicious elements
        for pattern, weight in SuppressionValidator.BANNER_INDICATORS:
            matches = list(re.finditer(pattern, html_content, re.IGNORECASE))
            for match in matches:
                banner_count += 1
                banner_score += weight
                
                # Extract context around match for size check
                start = max(0, match.start() - 500)
                end = min(len(html_content), match.end() + 500)
                context = html_content[start:end]
                
                # Check for size indicators in context
                width_match = re.search(r'width[=:]\s*["\']?(\d+)', context, re.IGNORECASE)
                height_match = re.search(r'height[=:]\s*["\']?(\d+)', context, re.IGNORECASE)
                
                if width_match and height_match:
                    width = int(width_match.group(1))
                    height = int(height_match.group(1))
                    if width * height > SuppressionValidator.SIZE_THRESHOLD:
                        # Large overlay detected, increase confidence
                        banner_score += 0.3
        
        # Cap confidence at 1.0
        confidence = min(1.0, banner_score)
        
        # Banner is present if we have reasonable confidence and found indicators
        is_present = confidence > 0.5 and banner_count > 0
        
        return is_present, confidence
    
    @staticmethod
    def should_retry(confidence: float, attempt: int, max_retries: int) -> bool:
        """
        Determine if a retry should be attempted based on confidence and attempt count.
        
        Args:
            confidence: Confidence score from validation (0.0-1.0)
            attempt: Current attempt number (1-indexed)
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if retry should be attempted
        """
        if attempt >= max_retries + 1:
            return False
        
        # Retry if confidence is above threshold
        return confidence > 0.5
