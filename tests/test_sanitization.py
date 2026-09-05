"""
Unit Tests for PII Sanitization Module
"""

import pytest
from src.data.sanitization import TextSanitizer, sanitize_text, sanitize_text_with_audit


@pytest.fixture
def sanitizer():
    return TextSanitizer()


def test_sanitize_empty_and_whitespace(sanitizer):
    """Verify empty, null, or whitespace-only strings return cleanly."""
    assert sanitizer.sanitize("") == ""
    assert sanitizer.sanitize("    ") == ""
    assert sanitizer.sanitize(None) == ""


def test_sanitize_emails(sanitizer):
    """Verify emails in various contexts are redacted with [EMAIL]."""
    text = "Reach me at test.user@example.com or support@subtext.org for help."
    cleaned, audit = sanitizer.sanitize_with_audit(text)
    assert "[EMAIL]" in cleaned
    assert "test.user@example.com" not in cleaned
    assert "support@subtext.org" not in cleaned
    assert audit.replacements["email"] == 2


def test_sanitize_urls(sanitizer):
    """Verify http, https, and short URLs are redacted with [URL]."""
    text = "Check out https://github.com/example/repo and visit http://crisis-support.org/help."
    cleaned, audit = sanitizer.sanitize_with_audit(text)
    assert "[URL]" in cleaned
    assert "https://github.com" not in cleaned
    assert "http://crisis-support.org" not in cleaned
    assert audit.replacements["url"] == 2


def test_sanitize_user_handles(sanitizer):
    """Verify Reddit u/ and Twitter @ handles are scrubbed."""
    text = "Thanks @alice_99 and u/bob-helper for the kind words!"
    cleaned, audit = sanitizer.sanitize_with_audit(text)
    assert "[USER]" in cleaned
    assert "@alice_99" not in cleaned
    assert "u/bob-helper" not in cleaned
    assert audit.replacements["user_handle"] == 2


def test_sanitize_subreddits(sanitizer):
    """Verify subreddit tags like r/depression or /r/Anxiety are redacted."""
    text = "I posted on r/depression and /r/mentalhealth earlier."
    cleaned, audit = sanitizer.sanitize_with_audit(text)
    assert "[SUBREDDIT]" in cleaned
    assert "r/depression" not in cleaned
    assert "/r/mentalhealth" not in cleaned
    assert audit.replacements["subreddit"] == 2


def test_sanitize_phone_numbers(sanitizer):
    """Verify standard phone numbers are redacted with [PHONE]."""
    text = "You can call 123-456-7890 or (800) 273-8255 anytime."
    cleaned, audit = sanitizer.sanitize_with_audit(text)
    assert "[PHONE]" in cleaned
    assert "123-456-7890" not in cleaned
    assert "(800) 273-8255" not in cleaned
    assert audit.replacements["phone"] >= 2


def test_sanitize_ipv4(sanitizer):
    """Verify IP addresses are scrubbed with [IP]."""
    text = "The server at 192.168.1.100 was logging connections."
    cleaned, audit = sanitizer.sanitize_with_audit(text)
    assert "[IP]" in cleaned
    assert "192.168.1.100" not in cleaned
    assert audit.replacements["ip"] == 1


def test_sanitize_html_entities_and_whitespace(sanitizer):
    """Verify HTML entities are properly unescaped and excess whitespace normalized."""
    text = "Me &amp; you &gt; them.   \n\n\n\nToo many newlines!"
    cleaned = sanitizer.sanitize(text)
    assert "Me & you > them." in cleaned
    assert "\n\nToo many" in cleaned
    assert "   " not in cleaned


def test_sanitize_emojis_and_unicode_preserved(sanitizer):
    """Verify that emotional indicators (emojis, punctuation) are preserved."""
    text = "I am so tired 😔💔 ... nothing makes sense anymore."
    cleaned = sanitizer.sanitize(text)
    assert "😔💔" in cleaned
    assert "nothing makes sense anymore." in cleaned


def test_sanitization_idempotency(sanitizer):
    """Verify that running sanitization twice yields identical results."""
    text = "Contact @john at john@doe.com or visit https://crisis.org! Call 555-123-4567."
    pass1 = sanitizer.sanitize(text)
    pass2 = sanitizer.sanitize(pass1)
    assert pass1 == pass2
