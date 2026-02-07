"""
Template Analyzer - Automatically identify static vs dynamic content in templates.

This is the core solution to the template setup problem:
- Takes a RAW template (firm's approved Word doc with no placeholders)
- Analyzes content to identify what's static vs dynamic
- Outputs a NEW template with {{placeholders}} inserted
- Generates prompt hints for LLM-generated sections

Reduces 4-hour manual setup to minutes of review.
"""

import re
import io
import zipfile
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class ContentType(Enum):
    """Classification of content sections"""
    STATIC = "static"           # Never changes (legal, headers, boilerplate)
    DYNAMIC_VALUE = "dynamic_value"     # Simple replacement (name, date, amount)
    DYNAMIC_LLM = "dynamic_llm"         # LLM-generated content (recommendations, assessments)
    TABLE_HEADER = "table_header"       # Table headers (usually static)
    TABLE_DATA = "table_data"           # Table data rows (usually dynamic)


@dataclass
class DynamicValue:
    """Represents a specific dynamic value found in the document"""
    original_text: str           # The actual text found (e.g., "John Smith")
    placeholder_name: str        # Generated placeholder name (e.g., "client_name")
    value_type: str             # Type: name, date, currency, percentage, etc.
    context: str                # Surrounding context (e.g., "Client Name:")
    confidence: float           # Detection confidence
    position: int               # Position in document


@dataclass
class AnalyzedSection:
    """Represents an analyzed section of the document"""
    text: str
    content_type: ContentType
    confidence: float
    placeholder_name: Optional[str] = None
    prompt_hint: Optional[str] = None
    reasoning: str = ""
    position: int = 0
    heading_context: Optional[str] = None
    value_type: Optional[str] = None


@dataclass
class AnalysisResult:
    """Complete analysis result for a template"""
    sections: List[AnalyzedSection] = field(default_factory=list)
    dynamic_values: List[DynamicValue] = field(default_factory=list)  # NEW: specific values to replace
    static_count: int = 0
    dynamic_value_count: int = 0
    dynamic_llm_count: int = 0
    overall_confidence: float = 0.0
    template_type: str = "unknown"

    def get_dynamic_sections(self) -> List[AnalyzedSection]:
        return [s for s in self.sections if s.content_type != ContentType.STATIC]

    def get_placeholders(self) -> Dict[str, str]:
        """Get mapping of placeholder names to prompt hints"""
        result = {}
        for s in self.sections:
            if s.placeholder_name:
                result[s.placeholder_name] = s.prompt_hint or s.text[:50]
        for v in self.dynamic_values:
            result[v.placeholder_name] = f"Extract {v.value_type} from input (example: {v.original_text})"
        return result


class TemplateAnalyzer:
    """
    Analyzes raw templates to identify static vs dynamic content.

    The core problem this solves:
    - Firm gives template with NO markers
    - We need to figure out what stays the same vs what changes per client
    - Output a template with proper placeholders
    """

    # Regex patterns for detecting dynamic values
    DYNAMIC_PATTERNS = {
        'currency': [
            (r'[£$€]\s*[\d,]+(?:\.\d{2})?', 'currency'),
            (r'(?:GBP|USD|EUR)\s*[\d,]+(?:\.\d{2})?', 'currency'),
        ],
        'percentage': [
            (r'[+-]?\d+\.?\d*\s*%', 'percentage'),
        ],
        'date': [
            (r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', 'date'),
            (r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}', 'date'),
            (r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}', 'date'),
        ],
        'email': [
            (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'email'),
        ],
        'phone': [
            (r'(?:\+44|0)\s*\d{2,4}\s*\d{3,4}\s*\d{3,4}', 'phone'),
            (r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}', 'phone'),
        ],
        'reference': [
            # Require word boundary and mandatory delimiter (: # or space+caps) to avoid matching inside words
            (r'\b(?:Policy|Account|Reference|Ref|SIPP|ISA)\s*[:\s#]\s*([A-Z][A-Z0-9\-]{2,20})', 'reference'),
        ],
    }

    # Context patterns that indicate a name follows
    # Using word boundary \b and stricter name matching (only proper names, not common words)
    NAME_CONTEXT_PATTERNS = [
        # Dear + title + name (e.g., "Dear Mr Smith", "Dear John")
        (r'\bDear\s+(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15}){0,2})\b', 'client_name'),
        # Client/Customer label + name (e.g., "Client: John Smith")
        (r'\b(?:Client|Customer)\s*:\s*([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15}){1,2})\b', 'client_name'),
        # Adviser/Author label + name (e.g., "Adviser: Jane Doe")
        (r'\b(?:Adviser|Advisor|Prepared by|Author)\s*:\s*([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15}){1,2})\b', 'adviser_name'),
    ]

    # Context patterns for specific value types
    VALUE_CONTEXT_PATTERNS = [
        (r'(?:pension|SIPP|personal pension)[^£$€\d]*([£$€]\s*[\d,]+(?:\.\d{2})?)', 'pension_value'),
        (r'(?:ISA|stocks and shares ISA)[^£$€\d]*([£$€]\s*[\d,]+(?:\.\d{2})?)', 'isa_value'),
        (r'(?:portfolio|investment|fund value|total value)[^£$€\d]*([£$€]\s*[\d,]+(?:\.\d{2})?)', 'portfolio_value'),
        (r'(?:transfer|transferring)[^£$€\d]*([£$€]\s*[\d,]+(?:\.\d{2})?)', 'transfer_amount'),
        (r'(?:contribution|investing)[^£$€\d]*([£$€]\s*[\d,]+(?:\.\d{2})?)', 'contribution_amount'),
    ]

    # Keywords indicating STATIC content (legal, compliance, boilerplate)
    STATIC_INDICATORS = {
        'legal': [
            'regulated by', 'fca', 'financial conduct authority',
            'authorised', 'authorized', 'disclaimer', 'terms and conditions',
            'privacy', 'data protection', 'gdpr', 'confidential',
            'copyright', 'all rights reserved', 'registered office',
            'company number', 'vat number', 'registered in england',
        ],
        'boilerplate': [
            'important information', 'risk warning', 'past performance',
            'capital at risk', 'tax treatment', 'legislation',
            'this document', 'for professional', 'not intended',
            'seek advice', 'independent advice',
        ],
    }

    # Keywords indicating DYNAMIC LLM content
    DYNAMIC_LLM_INDICATORS = [
        'recommendation', 'we recommend', 'our advice', 'suggest',
        'assessment', 'analysis', 'evaluation', 'review of',
        'rationale', 'reasoning', 'because', 'therefore',
        'your circumstances', 'your situation', 'your objectives',
    ]

    def __init__(self):
        self.current_heading = None
        self.placeholder_counter = {}

    def analyze_template(self, file_bytes: bytes) -> AnalysisResult:
        """
        Analyze a raw template to identify static vs dynamic content.
        """
        result = AnalysisResult()

        # Parse document
        document_xml, full_text = self._parse_document(file_bytes)

        # Detect template type
        result.template_type = self._detect_template_type(full_text)

        # STEP 1: Find all specific dynamic values (names, dates, amounts, etc.)
        result.dynamic_values = self._find_dynamic_values(full_text)

        # STEP 2: Analyze paragraphs for LLM sections
        paragraphs = self._extract_paragraphs(document_xml)

        position = 0
        self.current_heading = None

        for para_xml, para_text in paragraphs:
            if not para_text.strip():
                continue

            # Check if this is a heading
            is_heading = self._is_heading(para_xml, para_text)
            if is_heading:
                self.current_heading = para_text.strip()

            # Analyze the paragraph
            section = self._analyze_paragraph(para_text, para_xml, position)
            section.heading_context = self.current_heading

            result.sections.append(section)
            position += 1

            # Update counts
            if section.content_type == ContentType.STATIC:
                result.static_count += 1
            elif section.content_type == ContentType.DYNAMIC_LLM:
                result.dynamic_llm_count += 1

        # Count dynamic values
        result.dynamic_value_count = len(result.dynamic_values)

        # Calculate overall confidence
        all_confidences = [s.confidence for s in result.sections] + [v.confidence for v in result.dynamic_values]
        if all_confidences:
            result.overall_confidence = sum(all_confidences) / len(all_confidences)

        return result

    def _find_dynamic_values(self, full_text: str) -> List[DynamicValue]:
        """
        Find all specific dynamic values in the document.
        These are the actual values that need to be replaced with placeholders.
        """
        values = []
        seen_texts = set()

        # Find names with context
        for pattern, placeholder_base in self.NAME_CONTEXT_PATTERNS:
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                name = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if name and name not in seen_texts and len(name) > 3:
                    seen_texts.add(name)
                    placeholder = self._generate_unique_placeholder(placeholder_base)

                    # Get context
                    start = max(0, match.start() - 30)
                    context = full_text[start:match.start()].strip()

                    values.append(DynamicValue(
                        original_text=name,
                        placeholder_name=placeholder,
                        value_type='name',
                        context=context,
                        confidence=0.85,
                        position=match.start()
                    ))

        # Find values with specific context (pension values, ISA values, etc.)
        for pattern, placeholder_base in self.VALUE_CONTEXT_PATTERNS:
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                value = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if value and value not in seen_texts:
                    seen_texts.add(value)
                    placeholder = self._generate_unique_placeholder(placeholder_base)

                    start = max(0, match.start() - 50)
                    context = full_text[start:match.start()].strip()

                    values.append(DynamicValue(
                        original_text=value,
                        placeholder_name=placeholder,
                        value_type='currency',
                        context=context,
                        confidence=0.80,
                        position=match.start()
                    ))

        # Find generic dynamic patterns (dates, emails, phones, remaining currencies)
        for pattern_type, patterns in self.DYNAMIC_PATTERNS.items():
            for pattern, value_type in patterns:
                for match in re.finditer(pattern, full_text, re.IGNORECASE):
                    value = match.group(1).strip() if match.lastindex and match.lastindex >= 1 else match.group(0).strip()

                    if value and value not in seen_texts:
                        # Skip very short values (likely false positives)
                        if len(value) < 5 and pattern_type == 'reference':
                            continue

                        # Skip common English words that get falsely matched
                        common_words = {'value', 'start', 'will', 'over', 'ered', 'provider', 'remains'}
                        if value.lower() in common_words:
                            continue

                        # Check if this is in a static section (skip if so)
                        context_start = max(0, match.start() - 100)
                        context = full_text[context_start:match.start()].lower()

                        is_static_context = any(
                            indicator in context
                            for indicators in self.STATIC_INDICATORS.values()
                            for indicator in indicators
                        )

                        if not is_static_context:
                            seen_texts.add(value)
                            placeholder = self._generate_unique_placeholder(value_type)

                            values.append(DynamicValue(
                                original_text=value,
                                placeholder_name=placeholder,
                                value_type=value_type,
                                context=context[-50:].strip(),
                                confidence=0.75,
                                position=match.start()
                            ))

        # Sort by position
        values.sort(key=lambda v: v.position)

        return values

    def _generate_unique_placeholder(self, base_name: str) -> str:
        """Generate a unique placeholder name"""
        if base_name not in self.placeholder_counter:
            self.placeholder_counter[base_name] = 0
            return base_name
        else:
            self.placeholder_counter[base_name] += 1
            return f"{base_name}_{self.placeholder_counter[base_name]}"

    def generate_template(self, file_bytes: bytes, analysis: AnalysisResult = None) -> bytes:
        """
        Generate a new template with placeholders inserted for dynamic values.
        """
        if analysis is None:
            analysis = self.analyze_template(file_bytes)

        # Read original document
        template_io = io.BytesIO(file_bytes)
        output_io = io.BytesIO()

        with zipfile.ZipFile(template_io, 'r') as zin:
            with zipfile.ZipFile(output_io, 'w', zipfile.ZIP_DEFLATED) as zout:
                for item in zin.namelist():
                    data = zin.read(item)

                    if item == 'word/document.xml':
                        content = data.decode('utf-8')

                        # STEP 1: Replace specific dynamic values with placeholders
                        # Sort by length (longest first) to avoid partial replacements
                        sorted_values = sorted(analysis.dynamic_values,
                                              key=lambda v: len(v.original_text),
                                              reverse=True)

                        for dv in sorted_values:
                            placeholder = f"{{{{{dv.placeholder_name}}}}}"
                            # Replace in XML content
                            content = self._replace_in_xml(content, dv.original_text, placeholder)

                        # NOTE: LLM sections (large paragraphs) are NOT replaced automatically
                        # because they span multiple XML elements and would corrupt the document.
                        # They are identified in the analysis for manual handling.

                        data = content.encode('utf-8')

                    zout.writestr(item, data)

        return output_io.getvalue()

    def _replace_in_xml(self, xml_content: str, old_text: str, new_text: str) -> str:
        """
        Replace text ONLY within <w:t> tag content, not in XML structure.
        This prevents breaking XML tags when replacing text that appears in tag names.
        """
        # Escape the new text for XML
        escaped_new = self._escape_xml(new_text)

        # Only replace within <w:t>...</w:t> content
        result = []
        pos = 0

        # Find all <w:t> tags and their content
        t_start_pattern = re.compile(r'<w:t([^>]*)>')

        while pos < len(xml_content):
            # Find next <w:t> tag
            match = t_start_pattern.search(xml_content, pos)

            if not match:
                # No more <w:t> tags, append rest and done
                result.append(xml_content[pos:])
                break

            # Append everything before this <w:t> tag (unchanged)
            result.append(xml_content[pos:match.end()])

            # Find the closing </w:t>
            close_pos = xml_content.find('</w:t>', match.end())
            if close_pos == -1:
                # No closing tag, append rest and done
                result.append(xml_content[match.end():])
                break

            # Get the text content between <w:t> and </w:t>
            text_content = xml_content[match.end():close_pos]

            # Replace the old_text with new_text ONLY in this content
            if old_text in text_content:
                text_content = text_content.replace(old_text, escaped_new)

            result.append(text_content)
            result.append('</w:t>')

            pos = close_pos + 6  # len('</w:t>')

        return ''.join(result)

    def _extract_text_from_xml(self, xml: str) -> str:
        """Extract plain text from XML"""
        texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', xml)
        return ''.join(texts)

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters"""
        return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))

    def _analyze_paragraph(self, text: str, para_xml: str, position: int) -> AnalyzedSection:
        """Analyze a single paragraph to classify it"""
        text_lower = text.lower().strip()

        # Check for static indicators
        is_static = False
        for category, keywords in self.STATIC_INDICATORS.items():
            if any(kw in text_lower for kw in keywords):
                is_static = True
                break

        if is_static:
            return AnalyzedSection(
                text=text,
                content_type=ContentType.STATIC,
                confidence=0.90,
                reasoning="Contains legal/compliance/boilerplate keywords",
                position=position,
            )

        # Check for LLM content indicators
        is_llm = any(indicator in text_lower for indicator in self.DYNAMIC_LLM_INDICATORS)

        # Also check if paragraph is substantial and contains client-specific language
        has_client_refs = any(w in text_lower for w in ['you', 'your', 'client'])
        is_substantial = len(text) > 100

        if is_llm or (has_client_refs and is_substantial):
            placeholder_name = self._generate_section_placeholder(text)
            prompt_hint = self._generate_prompt_hint(text)

            return AnalyzedSection(
                text=text,
                content_type=ContentType.DYNAMIC_LLM,
                confidence=0.75 if is_llm else 0.60,
                placeholder_name=placeholder_name,
                prompt_hint=prompt_hint,
                reasoning="Contains recommendation/assessment language or client-specific content",
                position=position,
            )

        # Default to static for short or unclear content
        return AnalyzedSection(
            text=text,
            content_type=ContentType.STATIC,
            confidence=0.50,
            reasoning="No strong dynamic indicators found",
            position=position,
        )

    def _generate_section_placeholder(self, text: str) -> str:
        """Generate a placeholder name for an LLM section"""
        text_lower = text.lower()

        if 'recommend' in text_lower:
            return self._generate_unique_placeholder('recommendation')
        elif 'assess' in text_lower or 'risk' in text_lower:
            return self._generate_unique_placeholder('risk_assessment')
        elif 'rational' in text_lower or 'reason' in text_lower:
            return self._generate_unique_placeholder('rationale')
        elif 'summar' in text_lower or 'overview' in text_lower:
            return self._generate_unique_placeholder('summary')
        elif 'circumstance' in text_lower or 'situation' in text_lower:
            return self._generate_unique_placeholder('client_circumstances')
        elif 'objective' in text_lower or 'goal' in text_lower:
            return self._generate_unique_placeholder('objectives')
        else:
            return self._generate_unique_placeholder('dynamic_section')

    def _generate_prompt_hint(self, text: str) -> str:
        """Generate a hint for LLM prompt based on content"""
        text_lower = text.lower()

        if 'recommend' in text_lower:
            return "Generate personalized recommendations based on client circumstances and objectives."
        elif 'assess' in text_lower or 'risk' in text_lower:
            return "Provide risk assessment based on client's risk profile and capacity for loss."
        elif 'rational' in text_lower or 'reason' in text_lower:
            return "Explain the reasoning behind recommendations, linking to client's situation."
        elif 'summar' in text_lower or 'overview' in text_lower:
            return "Provide executive summary of key points and recommendations."
        elif 'circumstance' in text_lower or 'situation' in text_lower:
            return "Describe client's current financial circumstances and objectives."
        else:
            return f"Generate appropriate content. Original sample: {text[:100]}..."

    def generate_prompt_config(self, analysis: AnalysisResult) -> Dict:
        """Generate configuration for LLM prompts based on analysis."""
        config = {
            "template_type": analysis.template_type,
            "placeholders": {},
            "dynamic_values": [],
            "llm_sections": [],
        }

        # Add dynamic values
        for dv in analysis.dynamic_values:
            config["dynamic_values"].append({
                "placeholder": dv.placeholder_name,
                "type": dv.value_type,
                "example": dv.original_text,
                "context": dv.context,
            })
            config["placeholders"][dv.placeholder_name] = {
                "type": dv.value_type,
                "example": dv.original_text,
                "prompt": f"Extract {dv.value_type} from input data (example: {dv.original_text})",
            }

        # Add LLM sections
        for section in analysis.sections:
            if section.content_type == ContentType.DYNAMIC_LLM and section.placeholder_name:
                config["llm_sections"].append({
                    "placeholder": section.placeholder_name,
                    "prompt_hint": section.prompt_hint,
                    "heading_context": section.heading_context,
                    "sample": section.text[:200],
                })
                config["placeholders"][section.placeholder_name] = {
                    "type": "llm_generated",
                    "prompt": section.prompt_hint,
                }

        return config

    def _parse_document(self, file_bytes: bytes) -> Tuple[str, str]:
        """Parse Word document and return XML and full text"""
        document_xml = ""
        full_text = ""

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                if 'word/document.xml' in zf.namelist():
                    document_xml = zf.read('word/document.xml').decode('utf-8')
                    text_matches = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', document_xml)
                    full_text = ' '.join(text_matches)
        except Exception as e:
            raise ValueError(f"Failed to parse document: {e}")

        return document_xml, full_text

    def _extract_paragraphs(self, document_xml: str) -> List[Tuple[str, str]]:
        """Extract paragraphs with their XML and text"""
        paragraphs = []

        para_pattern = r'<w:p[^>]*>(.*?)</w:p>'
        for match in re.finditer(para_pattern, document_xml, re.DOTALL):
            para_xml = match.group(0)
            text_matches = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', para_xml)
            para_text = ''.join(text_matches)

            if para_text.strip():
                paragraphs.append((para_xml, para_text))

        return paragraphs

    def _is_heading(self, para_xml: str, para_text: str) -> bool:
        """Check if a paragraph is a heading"""
        if re.search(r'<w:pStyle w:val="Heading\d?"', para_xml):
            return True
        if ('<w:b/>' in para_xml or '<w:b ' in para_xml) and len(para_text.strip()) < 100:
            return True
        if para_text.isupper() and len(para_text.strip()) < 50:
            return True
        return False

    def _detect_template_type(self, full_text: str) -> str:
        """Detect the type of template"""
        text_lower = full_text.lower()

        if 'annual review' in text_lower or 'yearly review' in text_lower:
            return 'annual_review'
        elif 'suitability' in text_lower:
            return 'suitability_report'
        elif 'pension' in text_lower and 'transfer' in text_lower:
            return 'pension_transfer'
        elif 'investment' in text_lower and 'advice' in text_lower:
            return 'investment_advice'
        elif 'protection' in text_lower:
            return 'protection'
        elif 'fact find' in text_lower:
            return 'fact_find'
        else:
            return 'general'


def create_template_analyzer():
    """Factory function"""
    return TemplateAnalyzer()
