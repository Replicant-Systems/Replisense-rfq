import pytest
import os
from app.file_parser import FileParser, FileParsingError

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["attachment.docx", "attachment.pdf", "attachment.xlsx"])
async def test_parse_supported_file_types(filename):
    """Test supported file parsing for .docx, .pdf, .xlsx"""
    parser = FileParser()
    file_path = os.path.join(ASSETS_DIR, filename)
    result = await parser.parse_file_async(file_path)

    assert isinstance(result, dict), "Expected result to be a dictionary"
    assert result["source_file"] == filename
    assert result["file_size"] > 0
    assert "raw_text" in result and len(result["raw_text"]) > 10
    assert result["file_hash"]
    assert result["parsing_method"].startswith("async_")
    print(f"[TEST] ✅ Parsed {filename} successfully with method {result['parsing_method']}")

@pytest.mark.asyncio
async def test_parse_unsupported_extension(tmp_path):
    """Test that unsupported file types raise an appropriate error"""
    parser = FileParser()
    test_file = tmp_path / "test.xyz"
    test_file.write_text("unsupported content")

    with pytest.raises(FileParsingError, match="Unsupported file type"):
        await parser.parse_file_async(str(test_file))
    print(f"[TEST] ✅ Raised FileParsingError for unsupported format '.xyz'")

@pytest.mark.asyncio
async def test_file_not_found_error():
    """Test that missing files raise FileNotFoundError"""
    parser = FileParser()
    with pytest.raises(FileNotFoundError):
        await parser.parse_file_async("missing-file.docx")
    print("[TEST] ✅ Correctly raised FileNotFoundError for missing file")
