from __future__ import annotations

from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
READMES = (REPOSITORY_ROOT / "README.md", REPOSITORY_ROOT / "src/robot_agent/README.md")
FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
LOCAL_PATH = re.compile(
    r"(?<![\w/])((?:\./)?[\w.-]+\.sh|(?:src|tests)/[A-Za-z0-9_./*-]+)"
)
LOCAL_LINK = re.compile(r"\]\((?!https?://|mailto:|#)([^)]+)\)")


def referenced_local_paths(readme: Path) -> set[str]:
    text = readme.read_text(encoding="utf-8")
    fenced_text = "\n".join(FENCED_BLOCK.findall(text))
    references = {match.group(1).rstrip(".,:;`") for match in LOCAL_PATH.finditer(fenced_text)}
    references.update(match.group(1).split("#", 1)[0] for match in LOCAL_LINK.finditer(text))
    return {reference for reference in references if reference}


class ReadmePathTest(unittest.TestCase):
    def test_documented_local_paths_exist(self):
        missing: list[str] = []
        for readme in READMES:
            for reference in sorted(referenced_local_paths(readme)):
                normalized = reference.removeprefix("./").rstrip("/")
                if "*" in normalized:
                    exists = any(REPOSITORY_ROOT.glob(normalized))
                else:
                    exists = (REPOSITORY_ROOT / normalized).exists()
                if not exists:
                    missing.append(f"{readme.relative_to(REPOSITORY_ROOT)}: {reference}")

        self.assertEqual(missing, [], "README references missing local paths:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
