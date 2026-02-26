"""Test basic package import."""


def test_import_package():
    """Test that the package can be imported."""
    import thematic_analysis

    assert thematic_analysis.__version__ == "0.1.0"
