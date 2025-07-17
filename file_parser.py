# file_parser.py
from pathlib import Path
import pandas as pd
import PyPDF2
import docx

class FileParser:
    def parse_file(self, file_path: str) -> dict:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.txt':
            text = self._parse_txt(file_path)
        elif file_path.suffix.lower() == '.pdf':
            text = self._parse_pdf(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            text = self._parse_excel(file_path)
        elif file_path.suffix.lower() == '.docx':
            text = self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return {
            "raw_text": text,
            "source_file": str(file_path)
        }

    def _parse_txt(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _parse_pdf(self, file_path: Path) -> str:
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def _parse_excel(self, file_path: Path) -> str:
        text = ""
        df = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, sheet_df in df.items():
            text += f"Sheet: {sheet_name}\n"
            text += sheet_df.to_string(index=False) + "\n"
        return text

    def _parse_docx(self, file_path: Path) -> str:
        doc = docx.Document(file_path)
        return '\n'.join([p.text for p in doc.paragraphs])
