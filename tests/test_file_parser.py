import os
import pytest
from app.file_parser import FileParser

def test_parse_txt(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a sample RFQ")
    
    parser = FileParser()
    result = parser.parse_file(str(test_file))
    assert result["raw_text"] == "This is a sample RFQ"
    assert result["source_file"].endswith("test.txt")

def test_unsupported_format(tmp_path):
    test_file = tmp_path / "test.xyz"
    test_file.write_text("unknown format")

    parser = FileParser()
    with pytest.raises(ValueError):
        parser.parse_file(str(test_file))
