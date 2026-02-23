<MEDICAL_RECORD>
{medical_record}
</MEDICAL_RECORD>

Please extract structured medical information from the above record and return it in JSON format.
Use the following structure:
{{
  "symptoms": ["list", "of", "symptoms"],
  "diagnoses": ["list", "of", "diagnoses"],
  "medications": ["list", "of", "medications"],
  "tests": ["list", "of", "tests"],
  "uncertainties": ["list", "of", "uncertainties"],
  "risk_flags": ["list", "of", "risk_flags"]
}}

Return ONLY the JSON object, no additional text or explanation.
