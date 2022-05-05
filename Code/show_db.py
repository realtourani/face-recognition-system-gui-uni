import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets, QtSql




class BlobDelegate(QtWidgets.QStyledItemDelegate):
    def displayText(self, value, locale):
        if isinstance(value, QtCore.QByteArray):
            value = value.data().decode()
        return super(BlobDelegate, self).displayText(value, locale)


def createConnection():
    db = QtSql.QSqlDatabase.addDatabase("QSQLITE")
    db.setDatabaseName('Data/college.db')
    if not db.open():
        QtWidgets.QMessageBox.critical(
            None,
            QtWidgets.qApp.tr("Cannot open database"),
            QtWidgets.qApp.tr(
                "Unable to establish a database connection.\n"
                "This example needs SQLite support. Please read "
                "the Qt SQL driver documentation for information "
                "how to build it.\n\n"
                "Click Cancel to exit."
            ),
            QtWidgets.QMessageBox.Cancel,
        )
        return False
    return True


# if __name__ == "__main__":
import sys

# showdb_win = QtWidgets.QApplication(sys.argv)

if not createConnection():
    sys.exit(-1)

w = QtWidgets.QTableView()
w.horizontalHeader().setStretchLastSection(True)
w.setWordWrap(True)
w.setTextElideMode(QtCore.Qt.ElideLeft)
delegate = BlobDelegate(w)
w.setItemDelegateForColumn(4, delegate)
model = QtSql.QSqlQueryModel()
model.setQuery("SELECT * FROM faces")
w.setModel(model)
w.resize(640, 480)
w.show()

# showdb_win.exec_()