
import re
from email.utils import parseaddr
from urllib.parse import urlparse


class EmailRuleEngine:
    """
    Rule-based email security analysis.

    This engine identifies observable indicators.
    It does not independently prove that an email is malicious.
    """

    URGENCY_PATTERNS = [
        r"\burgent\b",
        r"\bimmediately\b",
        r"\bwithin\s+\d+\s+hours?\b",
        r"\bact\s+now\b",
        r"\bdeadline\b",
        r"\baccount\s+will\s+be\s+suspended\b",
    ]

    FINANCIAL_THREAT_PATTERNS = [
        r"\bsalary\s+(?:is\s+)?on\s+hold\b",
        r"\bsalary\s+processing\s+on\s+hold\b",
        r"\bpayment\s+(?:is\s+)?suspended\b",
        r"\baccount\s+will\s+be\s+blocked\b",
        r"\bpayment\s+will\s+be\s+delayed\b",
    ]

    SENSITIVE_REQUEST_PATTERNS = [
        r"\bpassword\b",
        r"\benter\s+your\s+credentials\b",
        r"\bverify\s+your\s+password\b",
        r"\bbank\s+account\s+details\b",
        r"\bcredit\s+card\s+details\b",
        r"\botp\b",
        r"\bone.time.password\b",
        r"\bsocial\s+security\s+number\b",
    ]

    ACCOUNT_VERIFICATION_PATTERNS = [
        r"\bverify\s+your\s+(?:employee\s+)?account\b",
        r"\bconfirm\s+your\s+account\b",
        r"\baccount\s+verification\b",
        r"\bverify\s+your\s+identity\b",
    ]

    ORGANIZATION_PATTERNS = [
        "payroll",
        "human resources",
        "hr department",
        "it support",
        "security team",
        "bank support",
        "technical support",
    ]

    FREE_EMAIL_DOMAINS = {
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "rediffmail.com",
        "icloud.com",
    }

    TRACKING_PARAMETERS = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "email_track_id",
        "open",
    }

    def __init__(self):
        self.results = []

    @staticmethod
    def _extract_urls(text):
        """Extract URLs from plain text or HTML email content."""

        pattern = r'https?://[^\s<>"\']+'

        urls = re.findall(pattern, text, flags=re.IGNORECASE)

        return list(dict.fromkeys(
            url.rstrip(".,;:!?)\"]}") for url in urls
        ))

    @staticmethod
    def _get_sender_domain(sender):
        """Extract sender email domain."""

        _, address = parseaddr(sender)

        if "@" not in address:
            return ""

        return address.rsplit("@", 1)[-1].lower()

    def _check_sender(self, sender, content):
        domain = self._get_sender_domain(sender)

        if not domain:
            return

        if domain in self.FREE_EMAIL_DOMAINS:
            claims_organization = any(
                phrase in content
                for phrase in self.ORGANIZATION_PATTERNS
            )

            if claims_organization:
                self.results.append({
                    "indicator": "Organizational identity mismatch",
                    "detail": (
                        f"The message claims an organizational role, "
                        f"but the sender uses a personal email domain: {domain}."
                    ),
                    "points": 2,
                })

    def _check_urgency(self, content):

        matches = [
            pattern
            for pattern in self.URGENCY_PATTERNS
            if re.search(pattern, content, re.IGNORECASE)
        ]

        if matches:
            self.results.append({
                "indicator": "Urgent language",
                "detail": "The email pressures the recipient to act quickly.",
                "points": 2,
            })

    def _check_financial_threats(self, content):

        matches = [
            pattern
            for pattern in self.FINANCIAL_THREAT_PATTERNS
            if re.search(pattern, content, re.IGNORECASE)
        ]

        if matches:
            self.results.append({
                "indicator": "Financial or account threat",
                "detail": (
                    "The email uses a salary, payment or account-related "
                    "threat to influence the recipient."
                ),
                "points": 2,
            })

    def _check_sensitive_requests(self, content):

        matches = [
            pattern
            for pattern in self.SENSITIVE_REQUEST_PATTERNS
            if re.search(pattern, content, re.IGNORECASE)
        ]

        if matches:
            self.results.append({
                "indicator": "Sensitive information request",
                "detail": (
                    "The email requests or references credentials, "
                    "financial information or authentication codes."
                ),
                "points": 4,
            })

    def _check_account_verification(self, content):

        matches = [
            pattern
            for pattern in self.ACCOUNT_VERIFICATION_PATTERNS
            if re.search(pattern, content, re.IGNORECASE)
        ]

        if matches:
            self.results.append({
                "indicator": "Account verification request",
                "detail": (
                    "The email asks the recipient to verify an account "
                    "or identity."
                ),
                "points": 2,
            })

    def _check_urls(self, content):

        urls = self._extract_urls(content)

        for url in urls:

            parsed = urlparse(url)
            domain = (parsed.hostname or "").lower()

            if not domain:
                continue

            if parsed.scheme.lower() == "http":
                self.results.append({
                    "indicator": "Unencrypted HTTP link",
                    "detail": (
                        f"The email contains an HTTP link: {domain}. "
                        "HTTPS is preferable for sensitive operations."
                    ),
                    "points": 1,
                })

            if domain == "example.com" or domain.endswith(".example"):
                self.results.append({
                    "indicator": "Placeholder URL",
                    "detail": (
                        f"{domain} is a reserved example domain. "
                        "This may indicate a simulated or incomplete link."
                    ),
                    "points": 0,
                })

            query_parameters = parsed.query.lower()

            has_tracking = any(
                parameter in query_parameters
                for parameter in self.TRACKING_PARAMETERS
            )

            if has_tracking:
                # Tracking alone is not considered a phishing indicator.
                continue

    def _check_authentication(self, headers):

        auth_results = headers.get(
            "authentication-results", ""
        ).lower()

        if not auth_results:
            self.results.append({
                "indicator": "Authentication results unavailable",
                "detail": (
                    "SPF, DKIM and DMARC results were not found "
                    "in the supplied Authentication-Results header."
                ),
                "points": 0,
            })
            return

        for mechanism in ("spf", "dkim", "dmarc"):

            match = re.search(
                rf"\b{mechanism}=(pass|fail|softfail|neutral|none|temperror|permerror)\b",
                auth_results,
            )

            if not match:
                self.results.append({
                    "indicator": f"{mechanism.upper()} result unavailable",
                    "detail": (
                        f"No {mechanism.upper()} result was found."
                    ),
                    "points": 0,
                })
                continue

            status = match.group(1)

            points = 2 if status in {"fail", "softfail"} else 0

            self.results.append({
                "indicator": f"{mechanism.upper()} {status.upper()}",
                "detail": (
                    f"Email authentication reports {mechanism.upper()} "
                    f"as {status.upper()}."
                ),
                "points": points,
            })

    def analyze(self, email):

        self.results = []

        sender = email.get("sender", "")
        subject = email.get("subject", "")
        body = email.get("body", "")
        headers = email.get("headers", {})

        content = f"{subject}\n{body}"

        self._check_sender(sender, content)
        self._check_urgency(content)
        self._check_financial_threats(content)
        self._check_sensitive_requests(content)
        self._check_account_verification(content)
        self._check_urls(content)
        self._check_authentication(headers)

        total_score = sum(
            item["points"] for item in self.results
        )

        # Avoid treating one isolated indicator as proof of phishing.
        strong_indicators = {
            "Sensitive information request",
            "Financial or account threat",
        }

        has_strong_indicator = any(
            item["indicator"] in strong_indicators
            for item in self.results
        )

        if total_score >= 6:
            risk_level = "high"
        elif total_score >= 3:
            risk_level = "medium"
        else:
            risk_level = "low"

        category = (
            "suspicious"
            if total_score >= 3
            else "normal"
        )

        return {
            "category": category,
            "risk_level": risk_level,
            "rule_score": total_score,
            "strong_indicator_present": has_strong_indicator,
            "indicators": self.results,
            "note": (
                "Rule-based findings are indicators, not proof "
                "of malicious intent."
            ),
        }
