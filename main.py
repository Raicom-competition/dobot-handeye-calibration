import sys

from PyQt5.QtWidgets import QApplication

from main_window import MainWindow


APP_STYLESHEET = """
QWidget#centralWidget {
    background-color: #F0F2F5;
}
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 4px;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #005A9E;
    font-weight: bold;
}
QLabel {
    color: #1F2937;
}
QLineEdit, QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 22px;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #005A9E;
}
QPushButton {
    background-color: #005A9E;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #00457D;
}
QPushButton:pressed {
    background-color: #00365F;
}
QPushButton#dangerButton {
    background-color: #DC2626;
}
QPushButton#dangerButton:hover {
    background-color: #B91C1C;
}
QTableWidget {
    background-color: #FFFFFF;
    alternate-background-color: #F8FAFC;
    gridline-color: #E5E7EB;
}
QHeaderView::section {
    background-color: #F1F5F9;
    color: #1F2937;
    padding: 4px;
    border: none;
    border-right: 1px solid #E5E7EB;
}
QWidget#statusBar {
    background-color: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-radius: 6px;
}
QPlainTextEdit {
    background-color: #0B1220;
    color: #D1D5DB;
    border: none;
    border-radius: 4px;
}
"""


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
