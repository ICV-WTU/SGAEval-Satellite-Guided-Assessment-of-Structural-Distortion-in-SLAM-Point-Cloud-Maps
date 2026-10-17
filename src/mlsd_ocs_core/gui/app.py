"""Launch the standalone MLSD picking and calculation page."""

import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from mlsd_ocs_core.gui.dialog import PcdAlignDialog


def main() -> None:
    app = QApplication(sys.argv)
    font = QFont(app.font())
    font.setPointSize(14)
    app.setFont(font)
    window = PcdAlignDialog()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
