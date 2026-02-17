import re
import json

class PhishingTranscriptAnalyzer:
    def __init__(self):
        # Database of Social Engineering Red Flags
        self.scam_indicators = {
            "urgency": [
                r"act now", r"immediate action", r"suspend your account", 
                r"expire in 24 hours", r"arrest warrant", r"locked out"
            ],
            "authority_abuse": [
                r"calling from the IRS", r"FBI investigation", r"Microsoft support", 
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
        print("[DeepGuard] NLP Phishing Detector Initialized.")

    def analyze_transcript_segment(self, text_segment):
        """
        Scans a live transcript text segment for scam patterns.
        Returns a risk score and identified triggers.
        """
        text = text_segment.lower()
        triggers = []
        score = 0
        
        for category, patterns in self.scam_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    triggers.append({
                        "category": category,
                        "phrase": pattern,
                        "context": f"Found '{pattern}' in text."
                    })
                    # Weighing logic
                    if category == "sensitive_requests":
                        score += 40
                    elif category == "financial_pressure":
                        score += 30
                    elif category == "urgency":
                        score += 20
                    else:
                        score += 15
        
        # Cap score at 100
        score = min(score, 100)
        
        return {
            "risk_score": score,
            "triggers": triggers,
            "is_scam_likely": score > 45
        }

if __name__ == "__main__":
    analyzer = PhishingTranscriptAnalyzer()
    
    # Test Cases
    test_phrases = [
        "Hello, this is Microsoft technical support. We detected a virus.",
        "You must refund the overpayment via gift cards immediately or we will suspend your account.",
        "Hey, how are you doing today?"
    ]
    
    print("Running NLP Analysis Checks...")
    for phrase in test_phrases:
        result = analyzer.analyze_transcript_segment(phrase)
        print(f"\nScanning: '{phrase}'")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Triggers: {json.dumps(result['triggers'], indent=2)}")
