import sys
import pytest


def main():
    sys.exit(pytest.main(["-m", "not llm and not integration and not cli"]))


if __name__ == "__main__":
    main()
