import re

class PhishingTranscriptAnalyzer:
    def __init__(self):
        # Database of Social Engineering Red Flags
        self.scam_indicators = {
            "urgency": [
                r"act now", r"immediate action", r"suspend your account", 
                r"expire in 24 hours", r"arrest warrant", r"locked out"
            ],
            "authority_abuse": [
                r"calling from the IRS", r"fbi investigation", r"microsoft support", 
                r"manager verification", r"legal action"
            ],
            "sensitive_requests": [
                r"confirm your ssn", r"social security number", r"credit card number",
                r"password", r"two-factor code", r"remote access", r"anydesk", r"teamviewer"
            ],
            "financial_pressure": [
                r"gift card", r"bitcoin", r"wire transfer", r"western union", 
                r"refund", r"overpayment"
            ]
        }

    def analyze_transcript(self, text):
        """
        Scans text for scam patterns. Returns anomalies and risk contribution.
        """
        text = text.lower()
        anomalies = []
        score = 0
        
        for category, patterns in self.scam_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    anomalies.append(f"Scam Trigger: '{pattern}' ({category})")
                    if category == "sensitive_requests": score += 40
                    elif category == "financial_pressure": score += 30
                    elif category == "urgency": score += 20
                    else: score += 15
        
        return min(score, 100), anomalies
