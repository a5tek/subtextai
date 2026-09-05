"""
Ethical PII Sanitization Module

Deterministic regex- and rule-based redaction of Personally Identifiable Information (PII)
and platform-specific handles before tokenization or embedding generation.
"""

import html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SanitizationAudit:
    """Audit record capturing PII transformations applied to an input text."""
    original_length: int = 0
    sanitized_length: int = 0
    replacements: Dict[str, int] = field(default_factory=dict)

    @property
    def has_modifications(self) -> bool:
        """Returns True if any PII pattern was detected and scrubbed."""
        return sum(self.replacements.values()) > 0


class TextSanitizer:
    """
    Deterministic text sanitizer implementing ethical PII scrubbing.
    
    Transforms sensitive identifiers (usernames, subreddit tags, URLs, emails,
    phone numbers, IP addresses) into standardized, semantic special tokens.
    Preserves linguistic context, emotional phrasing, and emojis.
    """

    # Replacement special tokens
    USER_TOKEN = "[USER]"
    SUBREDDIT_TOKEN = "[SUBREDDIT]"
    URL_TOKEN = "[URL]"
    EMAIL_TOKEN = "[EMAIL]"
    PHONE_TOKEN = "[PHONE]"
    IP_TOKEN = "[IP]"

    def __init__(self):
        # Pre-compile regexes for high-throughput matching
        self._email_regex = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        )
        
        # URLs: http, https, www, markdown links, common shorteners
        self._url_regex = re.compile(
            r"(?:https?://|www\.)[^\s<>\"]+|(?:\b[a-zA-Z0-9-]+\.(?:com|org|net|io|edu|gov|co|ly)\b[^\s<>\"]*)",
            re.IGNORECASE,
        )

        # Reddit & Twitter / social media handles
        # u/username, /u/username, @username
        self._user_handle_regex = re.compile(
            r"(?:(?<=^)|(?<=\s)|(?<=[(\[{]))(?:/?u/|@)[A-Za-z0-9_-]{2,30}\b",
            re.IGNORECASE,
        )

        # Subreddit tags: r/subreddit, /r/subreddit
        self._subreddit_regex = re.compile(
            r"(?:(?<=^)|(?<=\s)|(?<=[(\[{]))(?:/?r/)[A-Za-z0-9_-]{2,30}\b",
            re.IGNORECASE,
        )

        # Phone numbers: North American, international formats, crisis lines
        # Examples: 123-456-7890, (123) 456-7890, +1 800-273-8255, 123.456.7890
        self._phone_regex = re.compile(
            r"(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"
        )

        # IPv4 addresses
        self._ipv4_regex = re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        )

        # Excess whitespace / control characters (preserve normal unicode and emojis)
        self._control_chars_regex = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
        self._multi_whitespace_regex = re.compile(r"[ \t]+")
        self._multi_newline_regex = re.compile(r"\n{3,}")

    def sanitize(self, text: str) -> str:
        """
        Sanitizes text, returning only the cleaned string.
        
        Args:
            text: Raw input string.
            
        Returns:
            Sanitized string.
        """
        sanitized, _ = self.sanitize_with_audit(text)
        return sanitized

    def sanitize_with_audit(self, text: str) -> Tuple[str, SanitizationAudit]:
        """
        Sanitizes input text and returns both the cleaned string and an audit record.
        
        Args:
            text: Raw input string.
            
        Returns:
            Tuple of (sanitized_text, audit_record).
        """
        if not text or not isinstance(text, str):
            return "", SanitizationAudit(original_length=0, sanitized_length=0)

        original_len = len(text)
        counts: Dict[str, int] = {
            "email": 0,
            "url": 0,
            "user_handle": 0,
            "subreddit": 0,
            "phone": 0,
            "ip": 0,
        }

        # Step 1: Decode HTML entities (e.g. &amp; -> &, &lt; -> <)
        cleaned = html.unescape(text)

        # Step 2: Remove non-printable control characters
        cleaned = self._control_chars_regex.sub("", cleaned)

        # Step 3: Redact Email Addresses first (before general URL/handle splitting)
        cleaned, counts["email"] = self._email_regex.subn(self.EMAIL_TOKEN, cleaned)

        # Step 4: Redact URLs (must run before handles so domain names aren't split)
        cleaned, counts["url"] = self._url_regex.subn(self.URL_TOKEN, cleaned)

        # Step 5: Redact Subreddit handles
        cleaned, counts["subreddit"] = self._subreddit_regex.subn(self.SUBREDDIT_TOKEN, cleaned)

        # Step 6: Redact User handles
        cleaned, counts["user_handle"] = self._user_handle_regex.subn(self.USER_TOKEN, cleaned)

        # Step 7: Redact IPv4 addresses
        cleaned, counts["ip"] = self._ipv4_regex.subn(self.IP_TOKEN, cleaned)

        # Step 8: Redact Phone numbers
        cleaned, counts["phone"] = self._phone_regex.subn(self.PHONE_TOKEN, cleaned)

        # Step 9: Normalize multiple whitespaces while preserving natural single newlines
        cleaned = self._multi_whitespace_regex.sub(" ", cleaned)
        cleaned = self._multi_newline_regex.sub("\n\n", cleaned)
        cleaned = cleaned.strip()

        audit = SanitizationAudit(
            original_length=original_len,
            sanitized_length=len(cleaned),
            replacements=counts,
        )
        return cleaned, audit


# Module-level default singleton for easy imports
_DEFAULT_SANITIZER = TextSanitizer()


def sanitize_text(text: str) -> str:
    """Convenience helper to sanitize text using the default sanitizer instance."""
    return _DEFAULT_SANITIZER.sanitize(text)


def sanitize_text_with_audit(text: str) -> Tuple[str, SanitizationAudit]:
    """Convenience helper returning both sanitized text and audit report."""
    return _DEFAULT_SANITIZER.sanitize_with_audit(text)
