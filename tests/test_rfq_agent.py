import pytest
import os
import logging
from app.rfq_agent import RFQFieldGenerator
from app.file_parser import FileParser

# Setup logger
logger = logging.getLogger("rfq_agent_test")
logging.basicConfig(level=logging.INFO)

# Path to attachments
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
TEST_FILES = {
    "docx": os.path.join(ASSETS_DIR, "attachment.docx"),
    "xlsx": os.path.join(ASSETS_DIR, "attachment.xlsx"),
    "pdf": os.path.join(ASSETS_DIR, "attachment.pdf")
}

@pytest.mark.parametrize("file_type", ["docx", "xlsx", "pdf"])
def test_llm_extraction_from_attachment(file_type):
    """
    ✅ Production-grade test to evaluate LLM's ability to extract fields from real RFQ documents.
    Tests all supported formats (docx, xlsx, pdf).
    """
    file_path = TEST_FILES[file_type]
    logger.info(f" Testing LLM extraction from {file_type.upper()} file: {file_path}")

    parser = FileParser()
    agent = RFQFieldGenerator()

    try:
        # Parse text from actual file using file_parser
        text = asyncio_run(parser.parse_file_async(file_path))

        logger.info("📄 Parsed text from file successfully")
        logger.debug(f"Extracted Text Preview: {text}")

        # Run LLM agent on parsed text
        # result = agent.generate(text)
        result = agent.generate(text["raw_text"])

        # Log and assert result structure
        logger.info(f"📊 LLM response: Confidence={result.get('confidence_score')} | Success={result.get('success')}")
        assert result["success"] is True
        assert result["confidence_score"] > 0.7, "Low confidence score"
        assert isinstance(result.get("line_items", []), list), "Expected line_items list"

        logger.info(f"✅ {file_type.upper()} RFQ test passed with {len(result['line_items'])} items")

    except Exception as e:
        logger.exception(f"❌ Test failed for {file_type.upper()} file: {str(e)}")
        pytest.fail(f"Test failed for {file_type.upper()} due to exception")

# Helper for async call in sync pytest
import asyncio
def asyncio_run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)
